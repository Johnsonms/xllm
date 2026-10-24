"""Fused gated-product elementwise kernels (CuTe DSL, any arch), fp32 math, 16-byte vectors:

  swiglu_fwd(h1, h3)            -> h = silu(h1) * h3
  swiglu_bwd(h_grad, h1, h3)    -> (h1_grad, h3_grad) = (h_grad h3 silu'(h1), h_grad silu(h1))
  gate_mul / gate_bwd           -> the same with the gate silu or 'softplus2' (softplus with beta = ln 2, threshold 20:
                                   log2(1 + 2^x), derivative 1 / (1 + 2^-x)), for the gekko attention output gate

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


SOFTPLUS2_LINEAR = 20.0 / 0.6931471805599453  # beta x > 20: softplus is linear (torch's threshold)


class SwigluKernel:
    def __init__(self, dtype, backward: bool, gate: str = "silu", threads: int = 256):
        assert gate in ("silu", "softplus2")
        self.dtype, self.backward, self.gate, self.threads = dtype, backward, gate, threads

    @cute.jit
    def _gate(self, x: Float32):
        """(gate(x), gate'(x))."""
        val = Float32(0.0)
        der = Float32(0.0)
        if cutlass.const_expr(self.gate == "silu"):
            s = cute.arch.rcp_approx(1.0 + cute.math.exp2(-x * LOG2E, fastmath=True))
            val = x * s
            der = s * (1.0 + x * (1.0 - s))
        else:
            e = cute.math.exp2(x, fastmath=True)
            lin = x > SOFTPLUS2_LINEAR
            val = cutlass.select_(lin, x, cute.math.log2(1.0 + e, fastmath=True))
            der = cutlass.select_(lin, Float32(1.0), e * cute.arch.rcp_approx(1.0 + e))
        return val, der

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
                    val, _ = self._gate(h1[j])
                    o1[j] = val * h3[j]
            else:
                g, h1, h3 = ld(pA), ld(pB), ld(pC)
                o2 = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for j in cutlass.range_constexpr(VEC):
                    val, der = self._gate(h1[j])
                    o2[j] = g[j] * val
                    o1[j] = g[j] * h3[j] * der
                cute.make_tensor((pO2 + off).align(16), cute.make_layout(VEC)).store(o2.load().to(self.dtype))
            cute.make_tensor((pO1 + off).align(16), cute.make_layout(VEC)).store(o1.load().to(self.dtype))


_COMPILED = {}


def _fake(dtype):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=16)


def _run(backward, a, b, c, o1, o2, gate="silu"):
    key = (a.dtype, backward, gate)
    if key not in _COMPILED:
        dt = TORCH2CUTE[a.dtype]
        _COMPILED[key] = cute.compile(SwigluKernel(dt, backward, gate), *[_fake(dt) for _ in range(5)], Int32(0),
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


def gate_mul(x: torch.Tensor, m: torch.Tensor, gate: str) -> torch.Tensor:
    """gate(x) * m."""
    out = torch.empty_like(m)
    _run(False, x, m, m, out, out, gate)
    return out


def gate_bwd(g: torch.Tensor, x: torch.Tensor, m: torch.Tensor, gate: str):
    """Gradients of gate(x) * m: (x_grad, m_grad) = (g m gate'(x), g gate(x))."""
    x_grad, m_grad = torch.empty_like(x), torch.empty_like(m)
    _run(True, g, x, m, x_grad, m_grad, gate)
    return x_grad, m_grad
