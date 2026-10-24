"""Fused q / k preparation of the Gekko block (CuTe DSL, Blackwell): per head row of S channels

    forward:  y = rope(rmsnorm(x) * w), p = softmax(x)                 (one read of x, bf16 in / out, fp32 math)
    backward: dx = rmsnorm_bwd(rope_bwd(dy), x, w) + softmax_bwd(dp, softmax(x)) + extra, plus the partial sums
              of the norm weight gradient

replacing rmsnorm_fwd + apply_rope + F.softmax(dtype=fp32).to(bf16) (4 passes, one of them fp32) in the forward
and the recompute, _softmax_backward_data (fp32 round trips), rope backward, rmsnorm_bwd and two adds in the
backward. A warp handles one row (S / 32 contiguous channels per lane), 8 warps per CTA; a warp always sees
rows of one head so the weight-gradient partials stay in registers (partial[warp, S], summed on the host)."""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake

WARPS = 8


@cute.jit
def _warp_sum(v: Float32) -> Float32:
    for o in cutlass.range_constexpr(5):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << o)
    return v


@cute.jit
def _warp_max(v: Float32) -> Float32:
    for o in cutlass.range_constexpr(5):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=1 << o))
    return v


class QkPrepFwd:
    """y[row] = rope(bf16(rmsnorm(x[row]) * w[g])), pre[row] = bf16(rmsnorm(x[row]) * w[g]) (optional, before the
    rotation), p[row] = softmax(x[row]) (optional), rstd[row] (fp32, optional); rows = (b, l, g), x bf16 [rows, S].
    R channels are rotated with the (cos, sin) table of position l (R = 0: no rotation)."""

    def __init__(self, G: int, S: int, R: int, affine: bool, want_pre: bool, want_sm: bool):
        self.G, self.S, self.R, self.affine, self.want_pre, self.want_sm = G, S, R, affine, want_pre, want_sm
        self.E = S // 32
        assert S % 32 == 0 and self.E in (2, 4, 8, 16, 32) and R % self.E == 0 and 0 <= R <= S

    @cute.jit
    def __call__(self, tX: cute.Tensor, tF: cute.Tensor, tW: cute.Tensor, tY: cute.Tensor, tPre: cute.Tensor, tSm: cute.Tensor,
                 tRstd: cute.Tensor, rows: Int32, L: Int32, eps: Float32, stream: cuda.CUstream):
        self.kernel(tX.iterator, tF.iterator, tW.iterator, tY.iterator, tPre.iterator, tSm.iterator, tRstd.iterator, rows, L, eps).launch(
            grid=[(rows + WARPS - 1) // WARPS, 1, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pF: cute.Pointer, pW: cute.Pointer, pY: cute.Pointer, pPre: cute.Pointer, pSm: cute.Pointer,
               pRstd: cute.Pointer, rows: Int32, L: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        G, S, R, E = self.G, self.S, self.R, self.E
        lane = tidx % 32
        row = bidx * WARPS + tidx // 32
        if row < rows:
            g = row % G
            l = (row // G) % L
            off = Int64(row) * S + lane * E
            xv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            xv.store(cute.make_tensor((pX + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
            # rmsnorm
            ss = Float32(0.0)
            for k in cutlass.range_constexpr(E):
                ss = ss + xv[k] * xv[k]
            ss = _warp_sum(ss)
            rstd = cute.math.rsqrt(ss * (1.0 / S) + eps, fastmath=True)
            if lane == 0:
                cute.make_tensor(pRstd + row, cute.make_layout(1))[0] = rstd
            yv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            if cutlass.const_expr(self.affine):
                wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                wv.store(cute.make_tensor((pW + (g * S + lane * E)).align(2 * E), cute.make_layout(E)).load().to(Float32))
                for k in cutlass.range_constexpr(E):
                    yv[k] = xv[k] * rstd * wv[k]
            else:
                for k in cutlass.range_constexpr(E):
                    yv[k] = xv[k] * rstd
            yb = yv.load().to(BFloat16)  # the torch path rounds before the rotation
            if cutlass.const_expr(self.want_pre):
                cute.make_tensor((pPre + off).align(2 * E), cute.make_layout(E)).store(yb)
            if cutlass.const_expr(R > 0):
                if lane * E < R:
                    yv.store(yb.to(Float32))
                    fv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                    fv.store(cute.make_tensor((pF + (Int64(l) * R + lane * E)).align(4 * E), cute.make_layout(E)).load())
                    ov = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                    for i in cutlass.range_constexpr(E // 2):
                        x0, x1, c, s = yv[2 * i], yv[2 * i + 1], fv[2 * i], fv[2 * i + 1]
                        ov[2 * i] = x0 * c - x1 * s
                        ov[2 * i + 1] = x0 * s + x1 * c
                    cute.make_tensor((pY + off).align(2 * E), cute.make_layout(E)).store(ov.load().to(BFloat16))
                else:
                    cute.make_tensor((pY + off).align(2 * E), cute.make_layout(E)).store(yb)
            else:
                cute.make_tensor((pY + off).align(2 * E), cute.make_layout(E)).store(yb)
            # softmax
            if cutlass.const_expr(self.want_sm):
                m = Float32(-3.0e38)
                for k in cutlass.range_constexpr(E):
                    m = cute.arch.fmax(m, xv[k])
                m = _warp_max(m)
                ev = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                t = Float32(0.0)
                for k in cutlass.range_constexpr(E):
                    ev[k] = cute.math.exp(xv[k] - m, fastmath=True)
                    t = t + ev[k]
                t = _warp_sum(t)
                inv = 1.0 / t
                for k in cutlass.range_constexpr(E):
                    ev[k] = ev[k] * inv
                cute.make_tensor((pSm + off).align(2 * E), cute.make_layout(E)).store(ev.load().to(BFloat16))


class QkPrepBwd:
    """dx[row] = rmsnorm_bwd(rope_bwd(dy[row]), x[row], w[g]) + softmax_bwd(dp[row], softmax(x[row])) (optional)
    + extra[row] (optional); partial[warp, :] += rope_bwd(dy) * xhat for the weight gradient (affine; the warp's rows
    all belong to head warp % G). rstd is recomputed from x."""

    def __init__(self, G: int, S: int, R: int, affine: bool, has_sm: bool, has_extra: bool, n_warps: int):
        self.G, self.S, self.R, self.affine, self.has_sm, self.has_extra, self.n_warps = G, S, R, affine, has_sm, has_extra, n_warps
        self.E = S // 32
        assert S % 32 == 0 and self.E in (2, 4, 8, 16, 32) and R % self.E == 0 and 0 <= R <= S
        assert n_warps % WARPS == 0 and n_warps % G == 0

    @cute.jit
    def __call__(self, tDy: cute.Tensor, tDp: cute.Tensor, tEx: cute.Tensor, tX: cute.Tensor, tF: cute.Tensor, tW: cute.Tensor,
                 tDx: cute.Tensor, tPart: cute.Tensor, rows: Int32, L: Int32, eps: Float32, stream: cuda.CUstream):
        self.kernel(tDy.iterator, tDp.iterator, tEx.iterator, tX.iterator, tF.iterator, tW.iterator, tDx.iterator, tPart.iterator,
                    rows, L, eps).launch(grid=[self.n_warps // WARPS, 1, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDy: cute.Pointer, pDp: cute.Pointer, pEx: cute.Pointer, pX: cute.Pointer, pF: cute.Pointer, pW: cute.Pointer,
               pDx: cute.Pointer, pPart: cute.Pointer, rows: Int32, L: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        G, S, R, E = self.G, self.S, self.R, self.E
        lane = tidx % 32
        gw = bidx * WARPS + tidx // 32  # global warp: rows gw, gw + n_warps, ... (all of head gw % G)
        g = gw % G
        acc = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        wv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        for k in cutlass.range_constexpr(E):
            acc[k] = 0.0
            wv[k] = 1.0
        if cutlass.const_expr(self.affine):
            wv.store(cute.make_tensor((pW + (g * S + lane * E)).align(2 * E), cute.make_layout(E)).load().to(Float32))
        n_it = (rows - gw + self.n_warps - 1) // self.n_warps
        for it in cutlass.range(n_it):
            row = gw + it * self.n_warps
            l = (row // G) % L
            off = Int64(row) * S + lane * E
            xv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            xv.store(cute.make_tensor((pX + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
            gy = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            gy.store(cute.make_tensor((pDy + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
            # rope backward (conjugate rotation) of the norm-side gradient
            if cutlass.const_expr(R > 0):
                if lane * E < R:
                    fv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                    fv.store(cute.make_tensor((pF + (Int64(l) * R + lane * E)).align(4 * E), cute.make_layout(E)).load())
                    for i in cutlass.range_constexpr(E // 2):
                        x0, x1, c, s = gy[2 * i], gy[2 * i + 1], fv[2 * i], fv[2 * i + 1]
                        gy[2 * i] = x0 * c + x1 * s
                        gy[2 * i + 1] = x1 * c - x0 * s
            # rmsnorm backward: xhat = x rstd, y = xhat w; dxhat = gy w; dx = rstd (dxhat - xhat mean(dxhat xhat))
            ss = Float32(0.0)
            for k in cutlass.range_constexpr(E):
                ss = ss + xv[k] * xv[k]
            ss = _warp_sum(ss)
            rstd = cute.math.rsqrt(ss * (1.0 / S) + eps, fastmath=True)
            dot = Float32(0.0)
            for k in cutlass.range_constexpr(E):
                xh = xv[k] * rstd
                acc[k] = acc[k] + gy[k] * xh
                gy[k] = gy[k] * wv[k]
                dot = dot + gy[k] * xh
            dot = _warp_sum(dot) * (1.0 / S)
            dx = cute.make_rmem_tensor(cute.make_layout(E), Float32)
            for k in cutlass.range_constexpr(E):
                dx[k] = rstd * (gy[k] - xv[k] * rstd * dot)
            # softmax backward from the logits x
            if cutlass.const_expr(self.has_sm):
                gp = cute.make_rmem_tensor(cute.make_layout(E), Float32)
                gp.store(cute.make_tensor((pDp + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
                m = Float32(-3.0e38)
                for k in cutlass.range_constexpr(E):
                    m = cute.arch.fmax(m, xv[k])
                m = _warp_max(m)
                t = Float32(0.0)
                for k in cutlass.range_constexpr(E):
                    xv[k] = cute.math.exp(xv[k] - m, fastmath=True)
                    t = t + xv[k]
                t = _warp_sum(t)
                inv = 1.0 / t
                sdot = Float32(0.0)
                for k in cutlass.range_constexpr(E):
                    xv[k] = xv[k] * inv
                    sdot = sdot + xv[k] * gp[k]
                sdot = _warp_sum(sdot)
                for k in cutlass.range_constexpr(E):
                    dx[k] = dx[k] + xv[k] * (gp[k] - sdot)
            if cutlass.const_expr(self.has_extra):
                exv = cute.make_tensor((pEx + off).align(2 * E), cute.make_layout(E)).load().to(Float32)
                dx.store(dx.load() + exv)
            cute.make_tensor((pDx + off).align(2 * E), cute.make_layout(E)).store(dx.load().to(BFloat16))
        if cutlass.const_expr(self.affine):
            cute.make_tensor((pPart + (Int64(gw) * S + lane * E)).align(4 * E), cute.make_layout(E)).store(acc.load())


_NONE = {}


def _dummy(device):
    d = _NONE.get(device)
    if d is None:
        d = _NONE[device] = (torch.zeros(16, dtype=torch.bfloat16, device=device), torch.zeros(16, dtype=torch.float32, device=device))
    return d


def qk_prep_fwd(x: torch.Tensor, G: int, weight: Optional[torch.Tensor], freqs_cis: Optional[torch.Tensor], rope_dim: int,
                eps: float, want_pre: bool = False, want_sm: bool = True):
    """x [B, L, G * S] bf16 -> (y [B, L, G, S] = rope(rmsnorm(x) * (weight + 1)), pre [B, L, G, S] (before the rotation,
    or None), p [B, L, G, S] = softmax(x) (or None), rstd [B * L, G] fp32). weight: [G * S] or None (no affine)."""
    B, L, D = x.shape
    S = D // G
    R = rope_dim if freqs_cis is not None else 0
    x = x.contiguous()
    rows = B * L * G
    dev = x.device
    y = torch.empty(B, L, G, S, dtype=torch.bfloat16, device=dev)
    pre = torch.empty_like(y) if want_pre else None
    sm = torch.empty_like(y) if want_sm else None
    rstd = torch.empty(B * L, G, dtype=torch.float32, device=dev)
    db, df = _dummy(dev)
    f = torch.view_as_real(freqs_cis).contiguous().reshape(-1) if R > 0 else df
    w = (weight + 1.0).to(torch.bfloat16).reshape(-1) if weight is not None else db
    fn = _compiled(("qk_prep_fwd", G, S, R, weight is not None, want_pre, want_sm),
                   lambda: QkPrepFwd(G, S, R, weight is not None, want_pre, want_sm),
                   lambda: (_fake(BFloat16), _fake(Float32), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16),
                            _fake(Float32, 4), Int32(0), Int32(0), Float32(0.0)))
    fn(x.reshape(-1), f, w, y.reshape(-1), (pre if want_pre else y).reshape(-1), (sm if want_sm else y).reshape(-1), rstd.reshape(-1),
       rows, L, float(eps))
    return y, pre, sm, rstd


def qk_prep_bwd(dy: torch.Tensor, dp: Optional[torch.Tensor], extra: Optional[torch.Tensor], x: torch.Tensor, G: int,
                weight: Optional[torch.Tensor], freqs_cis: Optional[torch.Tensor], rope_dim: int, eps: float):
    """dx [B, L, G * S] bf16 (and the weight gradient [G * S] bf16 or None) of qk_prep_fwd: dy the gradient of y
    [B, L, G, S], dp of p (or None), extra an additional [B, L, G, S] term (or None); x the forward input."""
    B, L, D = x.shape
    S = D // G
    R = rope_dim if freqs_cis is not None else 0
    x = x.contiguous()
    rows = B * L * G
    dev = x.device
    n_warps = G * max(1, -(-int(os.environ.get("XLLM_QK_WARPS", "16384")) // G))
    n_warps = -(-n_warps // (WARPS * G)) * WARPS * G  # multiple of WARPS and G
    dx = torch.empty(B, L, D, dtype=torch.bfloat16, device=dev)
    affine = weight is not None
    part = torch.empty(n_warps, S, dtype=torch.float32, device=dev) if affine else None
    db, df = _dummy(dev)
    f = torch.view_as_real(freqs_cis).contiguous().reshape(-1) if R > 0 else df
    w = (weight + 1.0).to(torch.bfloat16).reshape(-1) if affine else db
    fn = _compiled(("qk_prep_bwd", G, S, R, affine, dp is not None, extra is not None, n_warps),
                   lambda: QkPrepBwd(G, S, R, affine, dp is not None, extra is not None, n_warps),
                   lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32), _fake(BFloat16),
                            _fake(BFloat16), _fake(Float32), Int32(0), Int32(0), Float32(0.0)))
    fn(dy.contiguous().reshape(-1), (dp.contiguous() if dp is not None else dy).reshape(-1),
       (extra.contiguous() if extra is not None else dy).reshape(-1), x.reshape(-1), f, w, dx.reshape(-1),
       (part if affine else df).reshape(-1), rows, L, float(eps))
    dw = part.view(n_warps // G, G, S).sum(0).reshape(-1).to(torch.bfloat16) if affine else None
    return dx, dw


def qk_prep_supported(x: torch.Tensor, G: int, rope_dim: int) -> bool:
    if os.environ.get("XLLM_CUTE_QK", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3:
        return False
    D = x.shape[-1]
    if D % G:
        return False
    S = D // G
    E = S // 32
    if S % 32 or E not in (2, 4, 8, 16, 32) or rope_dim < 0 or rope_dim > S or rope_dim % E or x.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 10
