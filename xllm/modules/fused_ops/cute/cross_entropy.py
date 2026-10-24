"""Cross-entropy rows of the fused linear cross entropy (CuTe DSL, any arch), model-parallel size 1:

  ce_stats:   per row of bf16 logits l [C, V]: m = max l, s = sum exp(l - m), loss = log s - (l[t] - m)
              (loss = 0 on ignored rows)
  ce_scores:  scores = exp(l - m) / s - onehot(t), 0 on ignored rows, written as fp32 (A of the TF32 grad_input
              GEMM) and bf16 (A of the grad_weight GEMM)

Replace .float() / max / sub / exp / sum / div / index / masked_fill and the scores.to(bf16) copy of
fused_blocks.cross_entropy (~11 passes over [C, V] fp32) by one read for the statistics and one read + two writes
for the scores. One CTA of 256 threads per row, 16-byte vectors.
"""
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
VEC = 8
THREADS = 256
NEG_BIG = -3.0e38


@cute.jit
def _block_reduce(v: Float32, sbuf: cute.Tensor, tidx: Int32, is_max: cutlass.Constexpr):
    """All-reduce over the CTA (8 warps): warp shuffles, then the 8 warp results through shared memory."""
    for o in cutlass.range_constexpr(5):
        w = cute.arch.shuffle_sync_bfly(v, offset=1 << o)
        v = cute.arch.fmax(v, w) if cutlass.const_expr(is_max) else v + w
    if tidx % 32 == 0:
        sbuf[tidx // 32] = v
    cute.arch.sync_threads()
    r = sbuf[0]
    for i in cutlass.range_constexpr(1, THREADS // 32):
        r = cute.arch.fmax(r, sbuf[i]) if cutlass.const_expr(is_max) else r + sbuf[i]
    cute.arch.sync_threads()
    return r


class CeRows:
    def __init__(self, V: int, scores: bool):
        assert V % VEC == 0
        self.V, self.scores = V, scores
        self.nvec = V // VEC

    @cute.jit
    def __call__(self, tL: cute.Tensor, tT: cute.Tensor, tIgn: cute.Tensor, tStats: cute.Tensor, tLoss: cute.Tensor,
                 tS32: cute.Tensor, tS16: cute.Tensor, rows: Int32, stream: cuda.CUstream):
        self.kernel(tL.iterator, tT.iterator, tIgn.iterator, tStats.iterator, tLoss.iterator, tS32.iterator,
                    tS16.iterator).launch(grid=[rows, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pL: cute.Pointer, pT: cute.Pointer, pIgn: cute.Pointer, pStats: cute.Pointer, pLoss: cute.Pointer,
               pS32: cute.Pointer, pS16: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sbuf = smem.allocate_tensor(Float32, cute.make_layout(THREADS // 32), byte_alignment=16)
        base = Int64(row) * self.V
        ign = cute.make_tensor(pIgn + row, cute.make_layout(1))[0] != 0
        t = Int32(cute.make_tensor(pT + row, cute.make_layout(1))[0])
        if cutlass.const_expr(not self.scores):
            # pass 1: online max / sum of exp over the row
            m = Float32(NEG_BIG)
            s = Float32(0.0)
            for i in cutlass.range(tidx, self.nvec, THREADS):
                v = cute.make_tensor((pL + (base + i * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                vm = v[0]
                for j in cutlass.range_constexpr(1, VEC):
                    vm = cute.arch.fmax(vm, v[j])
                mn = cute.arch.fmax(m, vm)
                acc = s * cute.math.exp2((m - mn) * LOG2E, fastmath=True)
                for j in cutlass.range_constexpr(VEC):
                    acc = acc + cute.math.exp2((v[j] - mn) * LOG2E, fastmath=True)
                m, s = mn, acc
            mrow = _block_reduce(m, sbuf, tidx, True)
            s = s * cute.math.exp2((m - mrow) * LOG2E, fastmath=True)
            srow = _block_reduce(s, sbuf, tidx, False)
            if tidx == 0:
                cute.make_tensor(pStats + 2 * row, cute.make_layout(1))[0] = mrow
                cute.make_tensor(pStats + 2 * row + 1, cute.make_layout(1))[0] = srow
                lt = cute.make_tensor(pL + (base + t), cute.make_layout(1))[0].to(Float32)
                loss = cute.math.log2(srow, fastmath=True) * LN2 - (lt - mrow)
                cute.make_tensor(pLoss + row, cute.make_layout(1))[0] = cutlass.select_(ign, Float32(0.0), loss)
        else:
            # pass 2: scores from the row statistics
            mrow = cute.make_tensor(pStats + 2 * row, cute.make_layout(1))[0]
            inv = cute.arch.rcp_approx(cute.make_tensor(pStats + 2 * row + 1, cute.make_layout(1))[0])
            keep = cutlass.select_(ign, Float32(0.0), Float32(1.0))
            for i in cutlass.range(tidx, self.nvec, THREADS):
                v = cute.make_tensor((pL + (base + i * VEC)).align(16), cute.make_layout(VEC)).load().to(Float32)
                o = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for j in cutlass.range_constexpr(VEC):
                    p = cute.math.exp2((v[j] - mrow) * LOG2E, fastmath=True) * inv
                    o[j] = (p - cutlass.select_(i * VEC + j == t, Float32(1.0), Float32(0.0))) * keep
                cute.make_tensor((pS32 + (base + i * VEC)).align(32), cute.make_layout(VEC)).store(o.load())
                cute.make_tensor((pS16 + (base + i * VEC)).align(16), cute.make_layout(VEC)).store(o.load().to(BFloat16))


_COMPILED = {}


def _fake(dtype, align):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def _compiled(V, scores):
    key = (V, scores)
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(
            CeRows(V, scores), _fake(BFloat16, 16), _fake(Int64, 8), _fake(Uint8, 1), _fake(Float32, 8),
            _fake(Float32, 4), _fake(Float32, 32), _fake(BFloat16, 16), Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    return _COMPILED[key]


def supported(logits: torch.Tensor) -> bool:
    return (logits.is_cuda and logits.dtype == torch.bfloat16 and logits.is_contiguous() and logits.dim() == 2
            and logits.shape[1] % VEC == 0 and logits.data_ptr() % 16 == 0)


def ce_rows(logits: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor, loss_out=None, compute_scores=False):
    """logits [C, V] bf16, target [C] int64 (any value on ignored rows), ignore [C] bool. Writes loss_out [C] fp32
    when given; returns (scores fp32, scores bf16) when compute_scores, else None."""
    C, V = logits.shape
    stats = torch.empty(C, 2, dtype=torch.float32, device=logits.device)
    loss = loss_out if loss_out is not None else torch.empty(C, dtype=torch.float32, device=logits.device)
    t = target.contiguous()
    ig = ignore.contiguous().view(torch.uint8)
    flat = logits.reshape(-1)
    _compiled(V, False)(flat, t, ig, stats.reshape(-1), loss, stats.reshape(-1), flat, C)
    if not compute_scores:
        return None
    s32 = torch.empty(C, V, dtype=torch.float32, device=logits.device)
    s16 = torch.empty(C, V, dtype=torch.bfloat16, device=logits.device)
    _compiled(V, True)(flat, t, ig, stats.reshape(-1), loss, s32.reshape(-1), s16.reshape(-1), C)
    return s32, s16
