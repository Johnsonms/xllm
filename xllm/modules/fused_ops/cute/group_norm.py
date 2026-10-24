"""Group RMS / layer norm forward (CuTe DSL, any arch): one warp per (row, group), vector loads, shuffle
reductions, several warps per block. Drop-in for xllm_extension group_rms_norm_fwd(_affine) /
group_layer_norm_fwd_affine, whose one-warp blocks with strided scalar loads run ~1.8x over the bandwidth floor.

  rms:   rstd = rsqrt(mean(x^2) + eps),  y = x rstd (w)
  layer: mean, rstd = rsqrt(var + eps),  y = (x - mean) rstd w + b
x [rows, G * CPG] contiguous (bf16 / fp16), CPG a multiple of 128 up to 1024; y like x, mean / rstd [rows, G] fp32.
"""
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64

TORCH2CUTE = {torch.bfloat16: BFloat16, torch.float16: Float16}


class GroupNormFwd:
    def __init__(self, dtype, CPG: int, G: int, layer: bool, affine: bool, warps: int = 8):
        assert CPG % 128 == 0 and CPG <= 1024 and (affine or not layer)
        self.dtype, self.CPG, self.G, self.layer, self.affine, self.warps = dtype, CPG, G, layer, affine, warps
        self.E = CPG // 32  # elements per lane
        self.VW = min(8, self.E)  # vector width
        self.NV = self.E // self.VW  # vectors per lane

    @cute.jit
    def __call__(self, tX: cute.Tensor, tW: cute.Tensor, tB: cute.Tensor, tY: cute.Tensor, tMean: cute.Tensor,
                 tRstd: cute.Tensor, rows: Int32, eps: Float32, stream: cuda.CUstream):
        n = rows * self.G
        grid = (n + self.warps - 1) // self.warps
        self.kernel(tX.iterator, tW.iterator, tB.iterator, tY.iterator, tMean.iterator, tRstd.iterator, n, eps).launch(
            grid=[grid, 1, 1], block=[32 * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pW: cute.Pointer, pB: cute.Pointer, pY: cute.Pointer, pMean: cute.Pointer,
               pRstd: cute.Pointer, n: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        item = bidx * self.warps + tidx // 32  # (row, group) index: row * G + group
        if item < n:
            g = item % self.G
            base = Int64(item) * self.CPG
            al = 2 * self.VW
            x = cute.make_rmem_tensor(cute.make_layout(self.E), Float32)
            for j in cutlass.range_constexpr(self.NV):
                v = cute.make_tensor((pX + (base + j * 32 * self.VW + lane * self.VW)).align(al),
                                     cute.make_layout(self.VW)).load().to(Float32)
                for e in cutlass.range_constexpr(self.VW):
                    x[j * self.VW + e] = v[e]
            mean = Float32(0.0)
            if cutlass.const_expr(self.layer):
                s = Float32(0.0)
                for i in cutlass.range_constexpr(self.E):
                    s = s + x[i]
                for o in cutlass.range_constexpr(5):
                    s = s + cute.arch.shuffle_sync_bfly(s, offset=1 << o)
                mean = s * (1.0 / self.CPG)
            q = Float32(0.0)
            for i in cutlass.range_constexpr(self.E):
                d = x[i] - mean
                q = q + d * d
            for o in cutlass.range_constexpr(5):
                q = q + cute.arch.shuffle_sync_bfly(q, offset=1 << o)
            rstd = cute.math.rsqrt(q * (1.0 / self.CPG) + eps)
            if lane == 0:
                cute.make_tensor(pRstd + item, cute.make_layout(1))[0] = rstd
                if cutlass.const_expr(self.layer):
                    cute.make_tensor(pMean + item, cute.make_layout(1))[0] = mean
            for j in cutlass.range_constexpr(self.NV):
                col = j * 32 * self.VW + lane * self.VW
                y = cute.make_rmem_tensor(cute.make_layout(self.VW), Float32)
                for e in cutlass.range_constexpr(self.VW):
                    y[e] = (x[j * self.VW + e] - mean) * rstd
                if cutlass.const_expr(self.affine):
                    w = cute.make_tensor((pW + (g * self.CPG + col)).align(al), cute.make_layout(self.VW)).load().to(Float32)
                    for e in cutlass.range_constexpr(self.VW):
                        y[e] = y[e] * w[e]
                    if cutlass.const_expr(self.layer):
                        bb = cute.make_tensor((pB + (g * self.CPG + col)).align(al), cute.make_layout(self.VW)).load().to(Float32)
                        for e in cutlass.range_constexpr(self.VW):
                            y[e] = y[e] + bb[e]
                cute.make_tensor((pY + (base + col)).align(al), cute.make_layout(self.VW)).store(y.load().to(self.dtype))


_COMPILED = {}


def _fake(dtype, align):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def supported(x: torch.Tensor, num_groups: int) -> bool:
    D = x.shape[-1]
    return (x.is_cuda and x.dtype in TORCH2CUTE and x.is_contiguous() and D % num_groups == 0
            and (D // num_groups) % 128 == 0 and D // num_groups <= 1024 and x.data_ptr() % 16 == 0)


def group_norm_fwd(x: torch.Tensor, num_groups: int, eps: float, weight=None, bias=None, layer: bool = False):
    """Returns (y, rstd) for rms, (y, mean, rstd) for layer norm; mean / rstd [rows, G] fp32."""
    D = x.shape[-1]
    G = num_groups
    CPG = D // G
    rows = x.numel() // D
    affine = weight is not None
    y = torch.empty_like(x)
    stat_shape = (rows, G)  # as the xllm_extension ops return them
    rstd = torch.empty(stat_shape, dtype=torch.float32, device=x.device)
    mean = torch.empty(stat_shape, dtype=torch.float32, device=x.device) if layer else rstd
    key = (x.dtype, CPG, G, layer, affine)
    if key not in _COMPILED:
        dt = TORCH2CUTE[x.dtype]
        al = 2 * min(8, CPG // 32)
        _COMPILED[key] = cute.compile(
            GroupNormFwd(dt, CPG, G, layer, affine), _fake(dt, al), _fake(dt, al), _fake(dt, al), _fake(dt, al),
            _fake(Float32, 4), _fake(Float32, 4), Int32(0), Float32(0.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    w = weight.contiguous() if affine else x
    b = bias.contiguous() if (layer and bias is not None) else x
    _COMPILED[key](x.reshape(-1), w.reshape(-1), b.reshape(-1), y.reshape(-1), mean.reshape(-1), rstd.reshape(-1), rows, eps)
    return (y, mean, rstd) if layer else (y, rstd)
