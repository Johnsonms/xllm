"""Fused rotary position embedding (CuTe DSL, any arch): y = rope(x) in one pass over x.

y[b, l, h, :R] = x[b, l, h, :R] rotated pair-wise by freqs_cis[l] (complex multiply, fp32, conjugated for the
backward), y[b, l, h, R:] = x[b, l, h, R:]. x is [B, L, H, D] with the last dim contiguous and any other strides
(it is not modified: partial-rope callers keep the input for the normalization backward); y is a new contiguous
tensor. Replaces the view_as_complex / multiply / type_as / cat chain of apply_rotary_embedding + apply_rope, which
makes about five passes over the data.
"""
from functools import lru_cache

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64

TORCH2CUTE = {torch.bfloat16: BFloat16, torch.float16: Float16}
VEC = 8  # 16 bytes of x per thread


class RopeKernel:
    def __init__(self, dtype, D: int, R: int, backward: bool, threads: int = 256):
        assert D % VEC == 0 and R % VEC == 0 and R <= D
        self.dtype, self.D, self.R, self.backward, self.threads = dtype, D, R, backward, threads
        self.vpr = D // VEC  # vectors per row
        self.rvpr = R // VEC  # rotated vectors per row

    @cute.jit
    def __call__(self, tX: cute.Tensor, tF: cute.Tensor, tY: cute.Tensor, rows: Int32, H: Int32, L: Int32,
                 sb: Int64, sl: Int64, sh: Int64, stream: cuda.CUstream):
        total = rows * self.vpr
        grid = (total + self.threads - 1) // self.threads
        self.kernel(tX.iterator, tF.iterator, tY.iterator, total, H, L, sb, sl, sh).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pF: cute.Pointer, pY: cute.Pointer, total: Int32, H: Int32, L: Int32,
               sb: Int64, sl: Int64, sh: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.threads + tidx
        if i < total:
            row = i // self.vpr
            vv = i % self.vpr
            h = row % H
            t = row // H
            l = t % L
            b = t // L
            src = Int64(b) * sb + Int64(l) * sl + Int64(h) * sh + vv * VEC
            x = cute.make_tensor((pX + src).align(16), cute.make_layout(VEC)).load()
            y = cute.make_rmem_tensor(cute.make_layout(VEC), self.dtype)
            if vv < self.rvpr:
                # (cos, sin) of the VEC / 2 pairs of this vector: freqs_cis[l, 4 vv .. 4 vv + 3] as fp32 pairs
                f = cute.make_tensor((pF + (Int64(l) * self.R + vv * VEC)).align(32), cute.make_layout(VEC)).load()
                xf = x.to(Float32)
                yf = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for p in cutlass.range_constexpr(VEC // 2):
                    a, c = xf[2 * p], xf[2 * p + 1]
                    cs, sn = f[2 * p], f[2 * p + 1]
                    if cutlass.const_expr(self.backward):
                        sn = -sn
                    yf[2 * p] = a * cs - c * sn
                    yf[2 * p + 1] = a * sn + c * cs
                y.store(yf.load().to(self.dtype))
            else:
                y.store(x)
            cute.make_tensor((pY + Int64(row) * self.D + vv * VEC).align(16), cute.make_layout(VEC)).store(y.load())


_COMPILED = {}


def _fake(dtype, align):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def rope(x: torch.Tensor, freqs_cis: torch.Tensor, rope_head_dim: int, backward: bool) -> torch.Tensor:
    """x [B, L, H, D] (last dim contiguous), freqs_cis [L, rope_head_dim / 2] complex64 -> new [B, L, H, D]."""
    B, L, H, D = x.shape
    R = rope_head_dim
    assert x.stride(3) == 1 and freqs_cis.shape == (L, R // 2)
    f = torch.view_as_real(freqs_cis.contiguous()).reshape(-1)
    y = torch.empty(B, L, H, D, dtype=x.dtype, device=x.device)
    key = (x.dtype, D, R, backward)
    if key not in _COMPILED:
        op = RopeKernel(TORCH2CUTE[x.dtype], D, R, backward)
        _COMPILED[key] = cute.compile(
            op, _fake(TORCH2CUTE[x.dtype], 16), _fake(Float32, 32), _fake(TORCH2CUTE[x.dtype], 16), Int32(0), Int32(0),
            Int32(0), Int64(0), Int64(0), Int64(0), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi")
    # flat view of x's storage from its first element (strides carry the layout)
    xs = x.as_strided((x.untyped_storage().nbytes() // x.element_size() - x.storage_offset(),), (1,))
    _COMPILED[key](xs, f, y.reshape(-1), B * L * H, H, L, x.stride(0), x.stride(1), x.stride(2))
    return y


@lru_cache(maxsize=None)
def supported(dtype: torch.dtype, D: int, R: int) -> bool:
    return dtype in TORCH2CUTE and D % VEC == 0 and R % VEC == 0 and 0 < R <= D
