"""Fused SwiGLU elementwise kernels (CuTe DSL, any arch), fp32 math, 16-byte vectors:

  swiglu_fwd(h1, h3)            -> h = silu(h1) * h3
  swiglu_bwd(h_grad, h1, h3)    -> (h1_grad, h3_grad) = (h_grad h3 silu'(h1), h_grad silu(h1))

Replace F.silu + mul (forward) and silu + mul + mul + mul + silu_backward (backward, dropout 0) of
fused_blocks.utils.swiglu_forward / _backward: 3 passes over [T, ffn] instead of 4 in the forward, 5 instead of
11 around the h_grad GEMM in the backward (h itself is still formed once for the fc2 weight gradient).
"""
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64

TORCH2CUTE = {torch.bfloat16: BFloat16, torch.float16: Float16}
VEC = 8
LOG2E = 1.4426950408889634


class SwigluKernel:
    def __init__(self, dtype, backward: bool, threads: int = 256):
        self.dtype, self.backward, self.threads = dtype, backward, threads

    @cute.jit
    def __call__(self, tA: cute.Tensor, tB: cute.Tensor, tC: cute.Tensor, tO1: cute.Tensor, tO2: cute.Tensor,
                 n_vec: Int32, stream: cuda.CUstream):
        grid = (n_vec + self.threads - 1) // self.threads
        self.kernel(tA.iterator, tB.iterator, tC.iterator, tO1.iterator, tO2.iterator, n_vec).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pA: cute.Pointer, pB: cute.Pointer, pC: cute.Pointer, pO1: cute.Pointer, pO2: cute.Pointer,
               n_vec: Int32):
        """fwd: A = h1, B = h3 -> O1 = h. bwd: A = h_grad, B = h1, C = h3 -> O1 = h1_grad, O2 = h3_grad."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * self.threads + tidx
        if i < n_vec:
            off = Int64(i) * VEC
            ld = lambda p: cute.make_tensor((p + off).align(16), cute.make_layout(VEC)).load().to(Float32)
            o1 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
            if cutlass.const_expr(not self.backward):
                h1, h3 = ld(pA), ld(pB)
                for j in cutlass.range_constexpr(VEC):
                    s = cute.arch.rcp_approx(1.0 + cute.math.exp2(-h1[j] * LOG2E, fastmath=True))
                    o1[j] = h1[j] * s * h3[j]
            else:
                g, h1, h3 = ld(pA), ld(pB), ld(pC)
                o2 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for j in cutlass.range_constexpr(VEC):
                    s = cute.arch.rcp_approx(1.0 + cute.math.exp2(-h1[j] * LOG2E, fastmath=True))
                    o2[j] = g[j] * h1[j] * s
                    o1[j] = g[j] * h3[j] * s * (1.0 + h1[j] * (1.0 - s))
                cute.make_tensor((pO2 + off).align(16), cute.make_layout(VEC)).store(o2.load().to(self.dtype))
            cute.make_tensor((pO1 + off).align(16), cute.make_layout(VEC)).store(o1.load().to(self.dtype))


_COMPILED = {}


def _fake(dtype):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=16)


def _run(backward, a, b, c, o1, o2):
    key = (a.dtype, backward)
    if key not in _COMPILED:
        dt = TORCH2CUTE[a.dtype]
        _COMPILED[key] = cute.compile(SwigluKernel(dt, backward), *[_fake(dt) for _ in range(5)], Int32(0),
                                      cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    _COMPILED[key](a.reshape(-1), b.reshape(-1), c.reshape(-1), o1.reshape(-1), o2.reshape(-1), a.numel() // VEC)


def supported(*ts) -> bool:
    t0 = ts[0]
    return all(t.is_cuda and t.dtype in TORCH2CUTE and t.dtype == t0.dtype and t.is_contiguous() and
               t.numel() == t0.numel() and t.data_ptr() % 16 == 0 for t in ts) and t0.numel() % VEC == 0


def swiglu_fwd(h1: torch.Tensor, h3: torch.Tensor) -> torch.Tensor:
    h = torch.empty_like(h1)
    _run(False, h1, h3, h3, h, h)
    return h


def swiglu_bwd(h_grad: torch.Tensor, h1: torch.Tensor, h3: torch.Tensor):
    h1_grad, h3_grad = torch.empty_like(h1), torch.empty_like(h3)
    _run(True, h_grad, h1, h3, h1_grad, h3_grad)
    return h1_grad, h3_grad
