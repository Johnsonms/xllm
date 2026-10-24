"""Depthwise causal conv1d (CuTe DSL, Blackwell), shared-memory staged: a CTA stages a tile of T timesteps (+ the
W - 1 halo rows) of CH channels with cp.async, then each thread walks the tile from shared memory with the last
W - 1 rows in registers. Forward: x tile; backward: x and out_grad tiles, dx and per-tile weight-gradient partials."""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.nvgpu import cpasync

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         "/home/johnson/xllm/xllm/modules/fused_ops/cute/adaptive_working_memory.py")
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake

VEC = 8
WIDTH = 4
HALO = WIDTH - 1


@cute.jit
def _silu(x: Float32) -> Float32:
    return x / (1.0 + cute.math.exp(-x, fastmath=True))


@cute.jit
def _dsilu(x: Float32) -> Float32:
    s = 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))
    return s * (1.0 + x * (1.0 - s))


class ConvFwdS:
    def __init__(self, act: bool, has_bos: bool, CH: int = 1024, T: int = 32):
        self.act, self.has_bos, self.CH, self.T = act, has_bos, CH, T
        self.threads = CH // VEC
        self.rows = T + HALO  # rows t0 - 3 .. t0 + T - 1

    @cute.jit
    def __call__(self, tX: cute.Tensor, tW: cute.Tensor, tBos: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, D: Int32,
                 stream: cuda.CUstream):
        self.kernel(tX.iterator, tW.iterator, tBos.iterator, tOut.iterator, L, D).launch(
            grid=[(D + self.CH - 1) // self.CH, (L + self.T - 1) // self.T, B], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pW: cute.Pointer, pBos: cute.Pointer, pOut: cute.Pointer, L: Int32, D: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, b = cute.arch.block_idx()
        CH, T, ROWS = self.CH, self.T, self.rows
        smem = utils.SmemAllocator()
        sX = smem.allocate_array(BFloat16, ROWS * CH, byte_alignment=128)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128)
        c = bx * CH + tidx * VEC
        t0 = by * T
        base = pX + Int64(b) * L * D + c
        valid_c = c < D
        # stage rows t0 - 3 .. t0 + T - 1 (zeros outside [0, L))
        for r in cutlass.range_constexpr(ROWS):
            t = t0 - HALO + r
            dst = cute.make_tensor((sX + (r * CH + tidx * VEC)).align(16), cute.make_layout(VEC))
            if valid_c & (t >= 0) & (t < L):
                cute.copy(atom, cute.make_tensor((base + Int64(t) * D).align(16), cute.make_layout(VEC)), dst)
            else:
                z = cute.make_rmem_tensor(cute.make_layout(VEC), BFloat16)
                for k in cutlass.range_constexpr(VEC):
                    z[k] = BFloat16(0.0)
                dst.store(z.load())
        cute.arch.cp_async_commit_group()
        w = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)
        if valid_c:
            for lag in cutlass.range_constexpr(WIDTH):
                wl = cute.make_tensor((pW + ((WIDTH - 1 - lag) * D + c)).align(32), cute.make_layout(VEC)).load()
                for k in cutlass.range_constexpr(VEC):
                    w[lag, k] = wl[k]
        # bos flags of rows t0 - 2 .. t0 + T - 1 -> registers via smem? read directly (L1 cached, CTA-uniform)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if valid_c:
            obase = pOut + Int64(b) * L * D + c
            bbase = pBos + Int64(b) * L
            xw = cute.make_rmem_tensor(cute.make_layout((HALO, VEC)), Float32)  # x[t - 1 - j]
            fl = cute.make_rmem_tensor(cute.make_layout(HALO), Float32)
            for j in cutlass.range_constexpr(HALO):
                xr = cute.make_tensor((sX + ((HALO - 1 - j) * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                for k in cutlass.range_constexpr(VEC):
                    xw[j, k] = xr[k]
                fl[j] = 0.0
                if cutlass.const_expr(self.has_bos):
                    tp = t0 - 1 - j
                    tpc = cutlass.select_(tp >= 0, tp, Int32(0))
                    fl[j] = cutlass.select_(tp >= 0, cute.make_tensor(bbase + tpc, cute.make_layout(1))[0].to(Float32), Float32(0.0))
            for i in cutlass.range_constexpr(T):
                t = t0 + i
                xt = cute.make_tensor((sX + ((HALO + i) * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                bt = Float32(0.0)
                if cutlass.const_expr(self.has_bos):
                    tt = cutlass.select_(t < L, t, L - 1)
                    bt = cutlass.select_(t < L, cute.make_tensor(bbase + tt, cute.make_layout(1))[0].to(Float32), Float32(0.0))
                v1 = 1.0 - bt
                v2 = v1 * (1.0 - fl[0])
                v3 = v2 * (1.0 - fl[1])
                o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for k in cutlass.range_constexpr(VEC):
                    acc = xt[k] * w[0, k] + xw[0, k] * w[1, k] * v1 + xw[1, k] * w[2, k] * v2 + xw[2, k] * w[3, k] * v3
                    o[k] = _silu(acc) if self.act else acc
                if t < L:
                    cute.make_tensor((obase + Int64(t) * D).align(16), cute.make_layout(VEC)).store(o.load().to(BFloat16))
                for k in cutlass.range_constexpr(VEC):
                    xw[2, k] = xw[1, k]
                    xw[1, k] = xw[0, k]
                    xw[0, k] = xt[k]
                fl[1] = fl[0]
                fl[0] = bt


class ConvBwdS:
    def __init__(self, act: bool, has_bos: bool, CH: int = 512, T: int = 32):
        self.act, self.has_bos, self.CH, self.T = act, has_bos, CH, T
        self.threads = CH // VEC
        self.xrows = T + 2 * HALO  # x rows t0 - 3 .. t0 + T + 2
        self.grows = T + HALO  # out_grad rows t0 .. t0 + T + 2

    @cute.jit
    def __call__(self, tG: cute.Tensor, tX: cute.Tensor, tW: cute.Tensor, tBos: cute.Tensor, tDx: cute.Tensor, tPart: cute.Tensor,
                 B: Int32, L: Int32, D: Int32, stream: cuda.CUstream):
        n_tiles = (L + self.T - 1) // self.T
        self.kernel(tG.iterator, tX.iterator, tW.iterator, tBos.iterator, tDx.iterator, tPart.iterator, L, D, n_tiles).launch(
            grid=[(D + self.CH - 1) // self.CH, n_tiles, B], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pG: cute.Pointer, pX: cute.Pointer, pW: cute.Pointer, pBos: cute.Pointer, pDx: cute.Pointer, pPart: cute.Pointer,
               L: Int32, D: Int32, n_tiles: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, b = cute.arch.block_idx()
        CH, T, XR, GR = self.CH, self.T, self.xrows, self.grows
        smem = utils.SmemAllocator()
        sX = smem.allocate_array(BFloat16, XR * CH, byte_alignment=128)
        sG = smem.allocate_array(BFloat16, GR * CH, byte_alignment=128)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128)
        c = bx * CH + tidx * VEC
        t0 = by * T
        valid_c = c < D
        base = pX + Int64(b) * L * D + c
        gbase = pG + Int64(b) * L * D + c
        for r in cutlass.range_constexpr(XR):
            t = t0 - HALO + r
            dst = cute.make_tensor((sX + (r * CH + tidx * VEC)).align(16), cute.make_layout(VEC))
            if valid_c & (t >= 0) & (t < L):
                cute.copy(atom, cute.make_tensor((base + Int64(t) * D).align(16), cute.make_layout(VEC)), dst)
            else:
                z = cute.make_rmem_tensor(cute.make_layout(VEC), BFloat16)
                for k in cutlass.range_constexpr(VEC):
                    z[k] = BFloat16(0.0)
                dst.store(z.load())
        for r in cutlass.range_constexpr(GR):
            t = t0 + r
            dst = cute.make_tensor((sG + (r * CH + tidx * VEC)).align(16), cute.make_layout(VEC))
            if valid_c & (t < L):
                cute.copy(atom, cute.make_tensor((gbase + Int64(t) * D).align(16), cute.make_layout(VEC)), dst)
            else:
                z = cute.make_rmem_tensor(cute.make_layout(VEC), BFloat16)
                for k in cutlass.range_constexpr(VEC):
                    z[k] = BFloat16(0.0)
                dst.store(z.load())
        cute.arch.cp_async_commit_group()
        w = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)
        dw = cute.make_rmem_tensor(cute.make_layout((WIDTH, VEC)), Float32)
        for lag in cutlass.range_constexpr(WIDTH):
            for k in cutlass.range_constexpr(VEC):
                dw[lag, k] = 0.0
        if valid_c:
            for lag in cutlass.range_constexpr(WIDTH):
                wl = cute.make_tensor((pW + ((WIDTH - 1 - lag) * D + c)).align(32), cute.make_layout(VEC)).load()
                for k in cutlass.range_constexpr(VEC):
                    w[lag, k] = wl[k]
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if valid_c:
            dbase = pDx + Int64(b) * L * D + c
            bbase = pBos + Int64(b) * L
            # pass 1: g[t] = out_grad[t] silu'(pre[t]) for rows t0 .. t0 + T + 2, written back over sG (bf16? keep fp32
            # precision: store g as bf16 would round; use the fp32 recompute per use instead) -> store g into sG as bf16
            # would change numerics vs the torch path (which keeps g in bf16 anyway: silu_backward output is bf16).
            xw = cute.make_rmem_tensor(cute.make_layout((HALO, VEC)), Float32)
            fl = cute.make_rmem_tensor(cute.make_layout(HALO), Float32)
            for j in cutlass.range_constexpr(HALO):
                xr = cute.make_tensor((sX + ((HALO - 1 - j) * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                for k in cutlass.range_constexpr(VEC):
                    xw[j, k] = xr[k]
                fl[j] = 0.0
                if cutlass.const_expr(self.has_bos):
                    tp = t0 - 1 - j
                    tpc = cutlass.select_(tp >= 0, tp, Int32(0))
                    fl[j] = cutlass.select_(tp >= 0, cute.make_tensor(bbase + tpc, cute.make_layout(1))[0].to(Float32), Float32(0.0))
            for i in cutlass.range_constexpr(GR):
                t = t0 + i
                xt = cute.make_tensor((sX + ((HALO + i) * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                gs = cute.make_tensor((sG + (i * CH + tidx * VEC)).align(16), cute.make_layout(VEC))
                go = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                go.store(gs.load().to(Float32))
                bt = Float32(0.0)
                if cutlass.const_expr(self.has_bos):
                    tt = cutlass.select_(t < L, t, L - 1)
                    bt = cutlass.select_(t < L, cute.make_tensor(bbase + tt, cute.make_layout(1))[0].to(Float32), Float32(0.0))
                v1 = 1.0 - bt
                v2 = v1 * (1.0 - fl[0])
                v3 = v2 * (1.0 - fl[1])
                g = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for k in cutlass.range_constexpr(VEC):
                    pre = xt[k] * w[0, k] + xw[0, k] * w[1, k] * v1 + xw[1, k] * w[2, k] * v2 + xw[2, k] * w[3, k] * v3
                    g[k] = go[k] * (_dsilu(pre) if self.act else Float32(1.0))
                if i < T:
                    for k in cutlass.range_constexpr(VEC):
                        dw[0, k] = dw[0, k] + g[k] * xt[k]
                        dw[1, k] = dw[1, k] + g[k] * xw[0, k] * v1
                        dw[2, k] = dw[2, k] + g[k] * xw[1, k] * v2
                        dw[3, k] = dw[3, k] + g[k] * xw[2, k] * v3
                gs.store(g.load().to(BFloat16))  # g (bf16, as the torch path's silu_backward output)
                for k in cutlass.range_constexpr(VEC):
                    xw[2, k] = xw[1, k]
                    xw[1, k] = xw[0, k]
                    xw[0, k] = xt[k]
                fl[1] = fl[0]
                fl[0] = bt
            # pass 2: dx[t] = g[t] w0 + g[t+1] w1 [bos[t+1]==0] + g[t+2] w2 [..] + g[t+3] w3 [..]
            for i in cutlass.range_constexpr(T):
                t = t0 + i
                if t < L:
                    f1 = Float32(0.0)
                    f2 = Float32(0.0)
                    f3 = Float32(0.0)
                    if cutlass.const_expr(self.has_bos):
                        t1 = cutlass.select_(t + 1 < L, t + 1, L - 1)
                        t2 = cutlass.select_(t + 2 < L, t + 2, L - 1)
                        t3 = cutlass.select_(t + 3 < L, t + 3, L - 1)
                        f1 = cutlass.select_(t + 1 < L, cute.make_tensor(bbase + t1, cute.make_layout(1))[0].to(Float32), Float32(0.0))
                        f2 = cutlass.select_(t + 2 < L, cute.make_tensor(bbase + t2, cute.make_layout(1))[0].to(Float32), Float32(0.0))
                        f3 = cutlass.select_(t + 3 < L, cute.make_tensor(bbase + t3, cute.make_layout(1))[0].to(Float32), Float32(0.0))
                    u1 = 1.0 - f1
                    u2 = u1 * (1.0 - f2)
                    u3 = u2 * (1.0 - f3)
                    g0 = cute.make_tensor((sG + (i * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                    g1 = cute.make_tensor((sG + ((i + 1) * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                    g2 = cute.make_tensor((sG + ((i + 2) * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                    g3 = cute.make_tensor((sG + ((i + 3) * CH + tidx * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                    dx = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                    for k in cutlass.range_constexpr(VEC):
                        dx[k] = g0[k] * w[0, k] + g1[k] * w[1, k] * u1 + g2[k] * w[2, k] * u2 + g3[k] * w[3, k] * u3
                    cute.make_tensor((dbase + Int64(t) * D).align(16), cute.make_layout(VEC)).store(dx.load().to(BFloat16))
            pp = pPart + (Int64(b * n_tiles + by) * WIDTH) * D + c
            for lag in cutlass.range_constexpr(WIDTH):
                o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for k in cutlass.range_constexpr(VEC):
                    o[k] = dw[lag, k]
                cute.make_tensor((pp + (WIDTH - 1 - lag) * D).align(32), cute.make_layout(VEC)).store(o.load())


def _bos_u8(bos_mask, x):
    if bos_mask is None:
        return torch.zeros(16, dtype=torch.uint8, device=x.device), False
    return bos_mask.to(torch.uint8).contiguous().reshape(-1), True


def conv_fwd(x, weight, bos_mask, activation, CH=1024, T=32):
    B, L, D = x.shape
    x = x.contiguous()
    w = weight.t().contiguous().to(torch.float32)
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    CH = min(CH, D)
    out = torch.empty_like(x)
    fn = _compiled(("conv_fwd_s", act, has_bos, CH, T), lambda: ConvFwdS(act, has_bos, CH, T), lambda: (
        _fake(BFloat16), _fake(Float32, 32), _fake(Uint8, 1), _fake(BFloat16), Int32(0), Int32(0), Int32(0)))
    fn(x.reshape(-1), w.reshape(-1), bos, out.reshape(-1), B, L, D)
    return out


def conv_bwd(out_grad, x, weight, bos_mask, activation, CH=512, T=32):
    B, L, D = x.shape
    x = x.contiguous()
    w = weight.t().contiguous().to(torch.float32)
    bos, has_bos = _bos_u8(bos_mask, x)
    act = activation in ("silu", "swish")
    CH = min(CH, D)
    n_tiles = (L + T - 1) // T
    dx = torch.empty_like(x)
    part = torch.empty(B * n_tiles, WIDTH, D, dtype=torch.float32, device=x.device)
    fn = _compiled(("conv_bwd_s", act, has_bos, CH, T), lambda: ConvBwdS(act, has_bos, CH, T), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(Float32, 32), _fake(Uint8, 1), _fake(BFloat16), _fake(Float32, 32), Int32(0), Int32(0), Int32(0)))
    fn(out_grad.contiguous().reshape(-1), x.reshape(-1), w.reshape(-1), bos, dx.reshape(-1), part.reshape(-1), B, L, D)
    return dx, part.sum(0).t().contiguous().to(weight.dtype)
