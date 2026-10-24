"""Fused SwiGLU elementwise ops (CuTe DSL, Blackwell), bf16 in / out with fp32 math and the torch path's roundings:

    forward:  hidden = bf16(bf16(silu(h1)) * h3)                                   (torch: 2 kernels, 5 passes -> 3)
    backward: h = hidden (for the w2 weight gradient), h1_grad = silu'(h1) bf16(h_grad h3), h3_grad = bf16(h_grad silu(h1))
                                                                                  (torch: 5 kernels, 14 passes -> 6)"""
import os
from typing import Tuple

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

THREADS = 256
VEC = 8


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))


class SwigluFwd:
    @cute.jit
    def __call__(self, tH1: cute.Tensor, tH3: cute.Tensor, tOut: cute.Tensor, n: Int32, stream: cuda.CUstream):
        n_thr = n // VEC
        self.kernel(tH1.iterator, tH3.iterator, tOut.iterator, n_thr).launch(
            grid=[(n_thr + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pH1: cute.Pointer, pH3: cute.Pointer, pOut: cute.Pointer, n_thr: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * THREADS + tidx
        if gid < n_thr:
            off = Int64(gid) * VEC
            h1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h1.store(cute.make_tensor((pH1 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            h3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h3.store(cute.make_tensor((pH3 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            for k in cutlass.range_constexpr(VEC):
                s = BFloat16(h1[k] * _sigmoid(h1[k])).to(Float32)
                o[k] = s * h3[k]
            cute.make_tensor((pOut + off).align(16), cute.make_layout(VEC)).store(o.load().to(BFloat16))


class SwigluBwd:
    @cute.jit
    def __call__(self, tH1: cute.Tensor, tH3: cute.Tensor, tG: cute.Tensor, tH: cute.Tensor, tD1: cute.Tensor, tD3: cute.Tensor,
                 n: Int32, stream: cuda.CUstream):
        n_thr = n // VEC
        self.kernel(tH1.iterator, tH3.iterator, tG.iterator, tH.iterator, tD1.iterator, tD3.iterator, n_thr).launch(
            grid=[(n_thr + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pH1: cute.Pointer, pH3: cute.Pointer, pG: cute.Pointer, pH: cute.Pointer, pD1: cute.Pointer, pD3: cute.Pointer,
               n_thr: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * THREADS + tidx
        if gid < n_thr:
            off = Int64(gid) * VEC
            h1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h1.store(cute.make_tensor((pH1 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            h3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            h3.store(cute.make_tensor((pH3 + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            g = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            g.store(cute.make_tensor((pG + off).align(16), cute.make_layout(VEC)).load().to(Float32))
            h = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            d1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            d3 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            for k in cutlass.range_constexpr(VEC):
                x = h1[k]
                sg = _sigmoid(x)
                s = BFloat16(x * sg).to(Float32)  # bf16 silu(h1) as in the torch path
                h[k] = s * h3[k]
                gs = BFloat16(g[k] * h3[k]).to(Float32)  # bf16 h1s_grad
                d1[k] = gs * (sg * (1.0 + x * (1.0 - sg)))
                d3[k] = g[k] * s
            cute.make_tensor((pH + off).align(16), cute.make_layout(VEC)).store(h.load().to(BFloat16))
            cute.make_tensor((pD1 + off).align(16), cute.make_layout(VEC)).store(d1.load().to(BFloat16))
            cute.make_tensor((pD3 + off).align(16), cute.make_layout(VEC)).store(d3.load().to(BFloat16))


def swiglu_fwd(h1: torch.Tensor, h3: torch.Tensor) -> torch.Tensor:
    """bf16(silu(h1)) * h3 (bf16)."""
    h1 = h1.contiguous()
    h3 = h3.contiguous()
    out = torch.empty_like(h1)
    fn = _compiled(("swiglu_fwd",), SwigluFwd, lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), Int32(0)))
    fn(h1.reshape(-1), h3.reshape(-1), out.reshape(-1), h1.numel())
    return out


def swiglu_bwd(h1: torch.Tensor, h3: torch.Tensor, h_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(hidden, h1_grad, h3_grad) of hidden = silu(h1) * h3 for the gradient h_grad (all bf16)."""
    h1 = h1.contiguous()
    h3 = h3.contiguous()
    h_grad = h_grad.contiguous()
    h = torch.empty_like(h1)
    d1 = torch.empty_like(h1)
    d3 = torch.empty_like(h1)
    fn = _compiled(("swiglu_bwd",), SwigluBwd, lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16),
                                                        _fake(BFloat16), Int32(0)))
    fn(h1.reshape(-1), h3.reshape(-1), h_grad.reshape(-1), h.reshape(-1), d1.reshape(-1), d3.reshape(-1), h1.numel())
    return h, d1, d3


def swiglu_supported(h1: torch.Tensor, dropout: float) -> bool:
    if os.environ.get("XLLM_CUTE_SWIGLU", "1") == "0" or h1.dtype != torch.bfloat16 or not h1.is_cuda or dropout != 0:
        return False
    if h1.numel() % VEC or h1.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(h1.device)[0] >= 9
