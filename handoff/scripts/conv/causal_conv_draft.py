"""Depthwise causal conv1d of the Gekko block (CuTe DSL, Blackwell): width-W taps per channel, segment resets from
the bos mask (a position only sees earlier positions of its own segment), optional silu; no bias / state.

    out[t, d] = act(sum_lag w[d, W - 1 - lag] x[t - lag, d] [no bos in (t - lag, t]])

A thread owns 8 channels and walks a tile of T timesteps with the last W - 1 rows in registers (one read of x, one
write of out); the backward recomputes the preactivation, keeps the last W - 1 gradients in registers (dx[t - 3]
once g[t] is known) and accumulates the weight-gradient partials per tile (summed on the host). The Triton kernels
run at ~1/3 of this (0.9B: 64 / 117 us per call -> ~20 / 35)."""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake

THREADS = 128
VEC = 4
T = 64
U = 8  # timesteps per loop iteration (independent loads in flight)
WIDTH = 4
assert T % U == 0 and U >= WIDTH - 1


@cute.jit
def _silu(x: Float32) -> Float32:
    return x / (1.0 + cute.math.exp(-x, fastmath=True))


@cute.jit
def _dsilu(x: Float32) -> Float32:
    s = 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))
    return s * (1.0 + x * (1.0 - s))


class ConvFwd:
    def __init__(self, act: bool, has_bos: bool):
        self.act, self.has_bos = act, has_bos

    @cute.jit
    def __call__(self, tX: cute.Tensor, tW: cute.Tensor, tBos: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, D: Int32,
                 stream: cuda.CUstream):
        self.kernel(tX.iterator, tW.iterator, tBos.iterator, tOut.iterator, L, D).launch(
            grid=[(D + THREADS * VEC - 1) // (THREADS * VEC), (L + T - 1) // T, B], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pW: cute.Pointer, pBos: cute.Pointer, pOut: cute.Pointer, L: Int32, D: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, b = cute.arch.block_idx()
        c = (bx * THREADS + tidx) * VEC
        if c < D:
            t0 = by * T
            base = pX + Int64(b) * L * D + c
            obase = pOut + Int64(b) * L * D + c
            bbase = pBos + Int64(b) * L
            w = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)  # w[lag, k] = weight[W - 1 - lag, c + k]
            for lag in cutlass.range_constexpr(WIDTH):
                wl = cute.make_tensor((pW + ((WIDTH - 1 - lag) * D + c)).align(4 * VEC), cute.make_layout(VEC)).load()
                for k in cutlass.range_constexpr(VEC):
                    w[lag, k] = wl[k]
            xw = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1, VEC)), Float32)  # xw[j] = x[t - 1 - j]
            flags = cute.make_rmem_tensor(cute.make_layout(WIDTH - 1), Float32)  # flags[j] = bos[t - 1 - j]
            for j in cutlass.range_constexpr(WIDTH - 1):
                flags[j] = 0.0
                tp = t0 - 1 - j
                if tp >= 0:
                    xr = cute.make_tensor((base + Int64(tp) * D).align(2 * VEC), cute.make_layout(VEC)).load().to(Float32)
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xr[k]
                    if cutlass.const_expr(self.has_bos):
                        flags[j] = cute.make_tensor(bbase + tp, cute.make_layout(1))[0].to(Float32)
                else:
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = 0.0
            for i4 in cutlass.range(T // U):
                tb = t0 + i4 * U
                xs = cute.make_rmem_tensor(cute.make_layout((U, VEC)), Float32)
                fb = cute.make_rmem_tensor(cute.make_layout(U), Float32)
                for u in cutlass.range_constexpr(U):
                    fb[u] = 0.0
                    if tb + u < L:
                        xr = cute.make_tensor((base + Int64(tb + u) * D).align(2 * VEC), cute.make_layout(VEC)).load().to(Float32)
                        for k in cutlass.range_constexpr(VEC):
                            xs[u, k] = xr[k]
                        if cutlass.const_expr(self.has_bos):
                            fb[u] = cute.make_tensor(bbase + (tb + u), cute.make_layout(1))[0].to(Float32)
                    else:
                        for k in cutlass.range_constexpr(VEC):
                            xs[u, k] = 0.0
                for u in cutlass.range_constexpr(U):
                    # rows t - 1, t - 2, t - 3 of row t = tb + u: from xs (u - 1 >= 0) or the window xw
                    f1 = fb[u - 1] if u >= 1 else flags[0]
                    f2 = fb[u - 2] if u >= 2 else flags[1 - u]
                    v1 = 1.0 - fb[u]
                    v2 = v1 * (1.0 - f1)
                    v3 = v2 * (1.0 - f2)
                    o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                    for k in cutlass.range_constexpr(VEC):
                        x1 = xs[u - 1, k] if u >= 1 else xw[0, k]
                        x2 = xs[u - 2, k] if u >= 2 else xw[1 - u, k]
                        x3 = xs[u - 3, k] if u >= 3 else xw[2 - u, k]
                        acc = xs[u, k] * w[0, k] + x1 * w[1, k] * v1 + x2 * w[2, k] * v2 + x3 * w[3, k] * v3
                        o[k] = _silu(acc) if self.act else acc
                    if tb + u < L:
                        cute.make_tensor((obase + Int64(tb + u) * D).align(2 * VEC), cute.make_layout(VEC)).store(o.load().to(BFloat16))
                for j in cutlass.range_constexpr(WIDTH - 1):
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xs[U - 1 - j, k]
                    flags[j] = fb[U - 1 - j]


class ConvBwd:
    def __init__(self, act: bool, has_bos: bool):
        self.act, self.has_bos = act, has_bos

    @cute.jit
    def __call__(self, tG: cute.Tensor, tX: cute.Tensor, tW: cute.Tensor, tBos: cute.Tensor, tDx: cute.Tensor, tPart: cute.Tensor,
                 B: Int32, L: Int32, D: Int32, stream: cuda.CUstream):
        n_tiles = (L + T - 1) // T
        self.kernel(tG.iterator, tX.iterator, tW.iterator, tBos.iterator, tDx.iterator, tPart.iterator, L, D, n_tiles).launch(
            grid=[(D + THREADS * VEC - 1) // (THREADS * VEC), n_tiles, B], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pG: cute.Pointer, pX: cute.Pointer, pW: cute.Pointer, pBos: cute.Pointer, pDx: cute.Pointer, pPart: cute.Pointer,
               L: Int32, D: Int32, n_tiles: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, b = cute.arch.block_idx()
        c = (bx * THREADS + tidx) * VEC
        if c < D:
            t0 = by * T
            base = pX + Int64(b) * L * D + c
            gbase = pG + Int64(b) * L * D + c
            dbase = pDx + Int64(b) * L * D + c
            bbase = pBos + Int64(b) * L
            w = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)
            for lag in cutlass.range_constexpr(WIDTH):
                wl = cute.make_tensor((pW + ((WIDTH - 1 - lag) * D + c)).align(4 * VEC), cute.make_layout(VEC)).load()
                for k in cutlass.range_constexpr(VEC):
                    w[lag, k] = wl[k]
            xw = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1, VEC)), Float32)  # x[t - 1 - j]
            gw = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1, VEC)), Float32)  # g[t - 1 - j]
            dw = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)
            flags = cute.make_rmem_tensor(cute.make_layout(WIDTH - 1), Float32)  # bos[t - 1 - j]
            for lag in cutlass.range_constexpr(WIDTH):
                for k in cutlass.range_constexpr(VEC):
                    dw[lag, k] = 0.0
            for j in cutlass.range_constexpr(WIDTH - 1):
                flags[j] = 0.0
                for k in cutlass.range_constexpr(VEC):
                    gw[j, k] = 0.0
                    xw[j, k] = 0.0
                tp = t0 - 1 - j
                if tp >= 0:
                    xr = cute.make_tensor((base + Int64(tp) * D).align(2 * VEC), cute.make_layout(VEC)).load().to(Float32)
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xr[k]
                    if cutlass.const_expr(self.has_bos):
                        flags[j] = cute.make_tensor(bbase + tp, cute.make_layout(1))[0].to(Float32)
            # rows t0 .. t0 + T + 2 (the last 3 are the halo: g only); dx[t - 3] once g[t] is known
            for i4 in cutlass.range((T + WIDTH - 1 + U - 1) // U):
                tb = t0 + i4 * U
                xs = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1 + U, VEC)), Float32)  # rows tb - 3 .. tb + 3
                gs = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1 + U, VEC)), Float32)
                fl = cute.make_rmem_tensor(cute.make_layout(WIDTH - 1 + U), Float32)
                gos = cute.make_rmem_tensor(cute.make_layout((U, VEC)), Float32)
                for j in cutlass.range_constexpr(WIDTH - 1):
                    fl[j] = flags[WIDTH - 2 - j]
                    for k in cutlass.range_constexpr(VEC):
                        xs[j, k] = xw[WIDTH - 2 - j, k]
                        gs[j, k] = gw[WIDTH - 2 - j, k]
                for u in cutlass.range_constexpr(U):
                    j = WIDTH - 1 + u
                    fl[j] = 0.0
                    for k in cutlass.range_constexpr(VEC):
                        xs[j, k] = 0.0
                        gos[u, k] = 0.0
                    if tb + u < L:
                        xr = cute.make_tensor((base + Int64(tb + u) * D).align(2 * VEC), cute.make_layout(VEC)).load().to(Float32)
                        gr = cute.make_tensor((gbase + Int64(tb + u) * D).align(2 * VEC), cute.make_layout(VEC)).load().to(Float32)
                        for k in cutlass.range_constexpr(VEC):
                            xs[j, k] = xr[k]
                            gos[u, k] = gr[k]
                        if cutlass.const_expr(self.has_bos):
                            fl[j] = cute.make_tensor(bbase + (tb + u), cute.make_layout(1))[0].to(Float32)
                for u in cutlass.range_constexpr(U):
                    j = WIDTH - 1 + u
                    v1 = 1.0 - fl[j]
                    v2 = v1 * (1.0 - fl[j - 1])
                    v3 = v2 * (1.0 - fl[j - 2])
                    in_tile = (i4 * U + u) < T
                    for k in cutlass.range_constexpr(VEC):
                        if cutlass.const_expr(self.act):
                            pre = xs[j, k] * w[0, k] + xs[j - 1, k] * w[1, k] * v1 + xs[j - 2, k] * w[2, k] * v2 + xs[j - 3, k] * w[3, k] * v3
                            gs[j, k] = gos[u, k] * _dsilu(pre)
                        else:
                            gs[j, k] = gos[u, k]
                    if in_tile:
                        for k in cutlass.range_constexpr(VEC):
                            dw[0, k] = dw[0, k] + gs[j, k] * xs[j, k]
                            dw[1, k] = dw[1, k] + gs[j, k] * xs[j - 1, k] * v1
                            dw[2, k] = dw[2, k] + gs[j, k] * xs[j - 2, k] * v2
                            dw[3, k] = dw[3, k] + gs[j, k] * xs[j - 3, k] * v3
                for u in cutlass.range_constexpr(U):
                    # dx of row ts = tb + u - 3 (local index u): g[ts] w0 + g[ts+1] w1 [..] + g[ts+2] w2 [..] + g[ts+3] w3 [..]
                    ts = tb + u - (WIDTH - 1)
                    if ts >= t0 and ts < L:
                        u1 = 1.0 - fl[u + 1]
                        u2 = u1 * (1.0 - fl[u + 2])
                        u3 = u2 * (1.0 - fl[u + 3])
                        dx = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                        for k in cutlass.range_constexpr(VEC):
                            dx[k] = gs[u, k] * w[0, k] + gs[u + 1, k] * w[1, k] * u1 + gs[u + 2, k] * w[2, k] * u2 + gs[u + 3, k] * w[3, k] * u3
                        cute.make_tensor((dbase + Int64(ts) * D).align(2 * VEC), cute.make_layout(VEC)).store(dx.load().to(BFloat16))
                for j in cutlass.range_constexpr(WIDTH - 1):
                    # window = the last 3 of the 7 rows: xw[0] = row tb + 3 (= xs[6]), xw[1] = xs[5], xw[2] = xs[4]
                    flags[j] = fl[WIDTH - 1 + U - 1 - j]
                    for k in cutlass.range_constexpr(VEC):
                        xw[j, k] = xs[WIDTH - 1 + U - 1 - j, k]
                        gw[j, k] = gs[WIDTH - 1 + U - 1 - j, k]
            pp = pPart + (Int64(b * n_tiles + by) * WIDTH) * D + c
            for lag in cutlass.range_constexpr(WIDTH):
                o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for k in cutlass.range_constexpr(VEC):
                    o[k] = dw[lag, k]
                cute.make_tensor((pp + (WIDTH - 1 - lag) * D).align(4 * VEC), cute.make_layout(VEC)).store(o.load())


def _bos_u8(bos_mask: Optional[torch.Tensor], x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if bos_mask is None:
        return torch.zeros(16, dtype=torch.uint8, device=x.device), False
    return bos_mask.to(torch.uint8).contiguous().reshape(-1), True


def conv_fwd(x: torch.Tensor, weight: torch.Tensor, bos_mask: Optional[torch.Tensor], activation: Optional[str]) -> torch.Tensor:
    """out [B, L, D] bf16 of the causal conv of x [B, L, D] (bf16) with weight [D, W] (any float dtype, W = 4)."""
    B, L, D = x.shape
    x = x.contiguous()
    w = weight.t().contiguous().to(torch.float32)  # [W, D]
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    out = torch.empty_like(x)
    fn = _compiled(("conv_fwd", act, has_bos), lambda: ConvFwd(act, has_bos), lambda: (
        _fake(BFloat16), _fake(Float32, 16), _fake(Uint8, 1), _fake(BFloat16), Int32(0), Int32(0), Int32(0)))
    fn(x.reshape(-1), w.reshape(-1), bos, out.reshape(-1), B, L, D)
    return out


def conv_bwd(out_grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bos_mask: Optional[torch.Tensor], activation: Optional[str]
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(dx [B, L, D] bf16, dweight [D, W] in weight's dtype) of conv_fwd."""
    B, L, D = x.shape
    x = x.contiguous()
    w = weight.t().contiguous().to(torch.float32)
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    n_tiles = (L + T - 1) // T
    dx = torch.empty_like(x)
    part = torch.empty(B * n_tiles, WIDTH, D, dtype=torch.float32, device=x.device)
    fn = _compiled(("conv_bwd", act, has_bos), lambda: ConvBwd(act, has_bos), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(Float32, 16), _fake(Uint8, 1), _fake(BFloat16), _fake(Float32, 16), Int32(0), Int32(0), Int32(0)))
    fn(out_grad.contiguous().reshape(-1), x.reshape(-1), w.reshape(-1), bos, dx.reshape(-1), part.reshape(-1), B, L, D)
    dweight = part.sum(0).t().contiguous().to(weight.dtype)
    return dx, dweight


def conv_supported(x: torch.Tensor, weight: torch.Tensor, bias, initial_state, output_final_state: bool, final_state_grad=None) -> bool:
    if os.environ.get("XLLM_CUTE_CONV", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3:
        return False
    if bias is not None or initial_state is not None or output_final_state or final_state_grad is not None:
        return False
    if weight.shape != (x.shape[-1], WIDTH) or x.shape[-1] % VEC or x.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 10
