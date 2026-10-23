"""Check the CuTe DSL adaptive working memory forward against an FP64 port of the reference algorithm.

Run from the repository root:
    python -m pytest tests/test_cute_adaptive_working_memory.py

The kernel module is loaded from its file so that the test does not need the compiled
xllm_extension that the xllm.modules package imports.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "xllm/modules/fused_ops/cute/adaptive_working_memory.py"
_spec = importlib.util.spec_from_file_location("cute_adaptive_working_memory", _PATH)
awm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(awm)


def reference_fwd(q, qk, k, v, C, memory, lnt, prev_qk, prev_k, prev_v, segment_idx, prev_segment_count):
    """FP64 port of xllm.modules.fused_ops.adaptive_working_memory._adaptive_working_memory_fwd (ortho=False):
    chunk by chunk, out_c = q_c M, then M <- M + r (kk (v - qk M) - M) with the previous chunk's keys."""
    B, H, L, S = q.shape
    K, V = v.shape[1], v.shape[3]
    nc, n_rep = L // C, H // K
    q, qk, k, v = (t.double() for t in (q, qk, k, v))
    has_prev = prev_k is not None
    prev_seg = None
    if segment_idx is not None and has_prev:
        prev_seg, segment_idx = segment_idx[:, :C], segment_idx[:, C:]
    out_mask = torch.zeros(B, nc, C, dtype=torch.bool, device=q.device)
    key_mask = memory_mask = prev_key_mask = None
    if segment_idx is not None:
        seg = segment_idx.view(B, nc, C)
        key_mask = (seg != seg[:, :, -1:]).view(B, 1, L, 1)
        memory_mask = torch.zeros(B, nc, dtype=torch.bool, device=q.device)
        memory_mask[:, 2:] = seg[:, :-2, -1] != seg[:, 1:-1, -1]
        out_mask[:, 2:] = seg[:, 2:] != seg[:, :-2, -1:]
        if has_prev:
            prev_key_mask = (prev_seg != prev_seg[:, -1:]).view(B, 1, C, 1)
            memory_mask[:, 0] = prev_segment_count != prev_seg[:, -1]
            memory_mask[:, 1] = prev_seg[:, -1] != seg[:, 0, -1]
            out_mask[:, 0] = seg[:, 0] != prev_segment_count.view(B, 1)
            out_mask[:, 1] = seg[:, 1] != prev_seg[:, -1:]

    def chunk_kk(kc, mask):  # softmax over time of a chunk's keys, and its log normalizer
        kc = kc.masked_fill(mask, float("-inf")) if mask is not None else kc
        return torch.softmax(kc, dim=2), torch.logsumexp(kc, dim=2)

    out = torch.zeros(B, H, L, V, dtype=torch.float64, device=q.device)
    if memory is None:
        M, lt = None, None
    else:
        M, lt = memory.double(), lnt.double()
    pq, pk, pv, pmask = (prev_qk.double(), prev_k.double(), prev_v.double(), prev_key_mask) if has_prev else (None,) * 4
    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        if M is None:
            out_mask[:, c] = True
            M = torch.zeros(B, K, S, V, dtype=torch.float64, device=q.device)
            lt = torch.full((B, K, S), float("-inf"), dtype=torch.float64, device=q.device)
        else:
            out[:, :, sl] = q[:, :, sl] @ M.repeat_interleave(n_rep, dim=1)
            out_mask[:, c] |= torch.isinf(lt[:, 0:1, 0])
            if memory_mask is not None:
                mm = memory_mask[:, c].view(B, 1, 1)
                lt = lt.masked_fill(mm, float("-inf"))
                M = M.masked_fill(mm.unsqueeze(-1), 0.0)
            kk, cur = chunk_kk(pk, pmask)
            new_lt = torch.logaddexp(cur, lt)
            r = torch.exp(cur - new_lt).unsqueeze(-1)
            M = M + r * (kk.transpose(2, 3) @ (pv - pq @ M) - M)
            lt = new_lt
        pq, pk, pv = qk[:, :, sl], k[:, :, sl], v[:, :, sl]
        pmask = key_mask[:, :, sl] if key_mask is not None else None
    out_mask = out_mask.view(B, L)
    return out.masked_fill(out_mask.view(B, 1, L, 1), 0.0), out_mask, M, lt


def make_inputs(B, H, K, S, V, L, C, with_memory, seg_rate, seed=0):
    """bf16 inputs as the fused block passes them: [B, H, L, S] views of [B, L, H, S] storage."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device="cuda", generator=g)
    view = lambda t: t.to(torch.bfloat16).transpose(1, 2)  # [B, L, H, D] storage -> [B, H, L, D] view
    q = view(torch.softmax(rn(B, L, H, S) * 2, -1))
    k_raw = rn(B, L, K, S) * 2
    k, qk, v = view(k_raw), view(torch.softmax(k_raw.to(torch.bfloat16).float(), -1)), view(rn(B, L, K, V))
    memory = lnt = pk = pqk = pv = None
    if with_memory:
        memory = (rn(B, K, S, V) * 0.1).to(torch.bfloat16)
        lnt = rn(B, K, S) + 8.0
        pk_raw = rn(B, C, K, S) * 2
        pk, pqk, pv = view(pk_raw), view(torch.softmax(pk_raw.to(torch.bfloat16).float(), -1)), view(rn(B, C, K, V))
    seg = cnt = None
    if seg_rate > 0:
        T = L + (C if with_memory else 0)
        seg = torch.cumsum((torch.rand(B, T, device="cuda", generator=g) < seg_rate / C).long(), dim=1) + 3
        if with_memory:
            cnt = seg[:, 0] - (torch.rand(B, device="cuda", generator=g) < 0.5).long()
    return q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt


def rel_err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


CASES = [  # B, H, K, S, V, L, C, with_memory, seg_rate
    (2, 4, 2, 128, 128, 1024, 256, False, 0.0),
    (2, 4, 2, 128, 128, 1024, 256, False, 0.3),
    (2, 4, 2, 128, 128, 1024, 256, True, 0.3),
    (1, 2, 1, 256, 512, 1536, 512, True, 0.0),
    (2, 4, 4, 128, 256, 512, 256, False, 0.5),  # two chunks: a single update
    (1, 4, 2, 256, 256, 2048, 512, True, 1.5),  # many segment boundaries
    (2, 8, 4, 256, 256, 16384, 4096, False, 0.3),  # gekko-0.9B
    (1, 4, 4, 256, 2048, 16384, 4096, False, 0.3),  # gekko-7.1B (tcgen05 recurrence)
    (1, 2, 2, 256, 1024, 4096, 1024, True, 0.5),  # tcgen05 recurrence with memory, previous chunk, segments
]


@pytest.mark.parametrize("B,H,K,S,V,L,C,with_memory,seg_rate", CASES)
def test_forward_matches_reference(B, H, K, S, V, L, C, with_memory, seg_rate):
    args = make_inputs(B, H, K, S, V, L, C, with_memory, seg_rate)
    out, mask, memory, lnt = awm.adaptive_working_memory_fwd(*args[:4], C, *args[4:])
    rout, rmask, rmemory, rlnt = reference_fwd(*args[:4], C, *args[4:])
    assert torch.equal(mask, rmask)
    # bf16 inputs and outputs; the reference implementation itself is a few 1e-3 off FP64 at these sizes
    assert rel_err(out, rout) < 1.5e-2
    assert rel_err(memory, rmemory) < 1.5e-2
    assert rel_err(lnt, rlnt) < 1e-5
    assert out.shape == (B, H, L, V) and out.transpose(1, 2).is_contiguous()


_REF_PATH = Path(__file__).resolve().parents[1] / "xllm/modules/fused_ops/cute/adaptive_working_memory_reference.py"
_ref_spec = importlib.util.spec_from_file_location("cute_awm_reference", _REF_PATH)
awm_ref = importlib.util.module_from_spec(_ref_spec)
_ref_spec.loader.exec_module(awm_ref)


@pytest.mark.parametrize("B,H,K,S,V,L,C,seg_rate", [
    (2, 4, 2, 16, 24, 256, 64, 0.0),
    (2, 4, 2, 16, 24, 384, 64, 0.6),
    (1, 2, 1, 32, 48, 512, 128, 1.0),
])
def test_phased_backward_matches_autograd(B, H, K, S, V, L, C, seg_rate):
    """The phase-decomposed backward (the plan of the backward kernels) against autograd, in FP64."""
    torch.manual_seed(0)
    f64 = dict(device="cuda", dtype=torch.float64)
    q = torch.softmax(torch.randn(B, H, L, S, **f64), -1).requires_grad_()
    k_raw = torch.randn(B, K, L, S, **f64) * 2
    k = k_raw.clone().requires_grad_()
    qk = torch.softmax(k_raw, -1).requires_grad_()
    v = torch.randn(B, K, L, V, **f64).requires_grad_()
    seg = (torch.cumsum((torch.rand(B, L, device="cuda") < seg_rate / C).long(), 1) + 3) if seg_rate > 0 else None
    out, _, memory, lnt = reference_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
    dout, dmem, dlnt = torch.randn_like(out), torch.randn_like(memory), torch.randn_like(lnt)
    ((out * dout).sum() + (memory * dmem).sum() + (lnt * dlnt).sum()).backward()
    with torch.no_grad():
        out2, memory2, _, saved = awm_ref.awm_fwd_saved(q, qk, k, v, C, None, None, None, None, None, seg, None)
        grads = awm_ref.awm_bwd_phased(dout, dmem, dlnt, q, qk, k, v, C, saved)
    assert rel_err(out2, out.detach()) < 1e-12 and rel_err(memory2, memory.detach()) < 1e-12
    for got, leaf in zip(grads, (q, qk, k, v)):
        assert rel_err(got, leaf.grad) < 1e-12
