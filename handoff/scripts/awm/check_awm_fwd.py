"""End-to-end AWM forward: CuTe vs the xllm reference (both bf16) against the fp64 phased prototype."""
import os
import sys
import types

S_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S_DIR, "xext"))
sys.path.insert(0, "/home/johnson/xllm")
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
import torch  # noqa: E402
import xllm.modules.fused_ops  # noqa: E402,F401
ref_mod = sys.modules["xllm.modules.fused_ops.adaptive_working_memory"]
from awm_proto import awm_fwd_phased  # noqa: E402
from check_proto import make  # noqa: E402
import adaptive_working_memory as cute_awm  # noqa: E402


def as_blhs_views(q, qk, k, v):
    """Inputs as the fused block passes them: [B, H, L, S] views of [B, L, H, S] storage."""
    return [t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, qk, k, v)]


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def check(B, H, K, S, V, L, C, mem, segf, seed=0):
    args = list(make(B, H, K, S, V, L, C, mem, segf, seed=seed, dtype=torch.bfloat16))
    args[:4] = as_blhs_views(*args[:4])
    q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
    if mem:
        pqk, pk, pv = as_blhs_views(pqk, pk, pv, pv)[:3]
    gold = awm_fwd_phased(q, qk, k, v, C, memory, lnt, pqk, pk, pv, seg, cnt)
    ref = ref_mod._adaptive_working_memory_fwd(q, qk, k, v, C, memory, lnt, pqk, pk, pv, seg, cnt)
    got = cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, memory, lnt, pqk, pk, pv, seg, cnt)
    torch.cuda.synchronize()
    names = ["out", "M", "lnt"]
    errs = [(rel(got[i], gold[i]), rel(ref[i], gold[i])) for i in (0, 2, 3)]
    mask_ok = torch.equal(got[1], gold[1]) and torch.equal(ref[1], gold[1])
    print(f"B{B} H{H} K{K} S{S} V{V} L{L} C{C} mem={mem} seg={segf}: mask_ok={mask_ok} " +
          " ".join(f"{n} cute {e[0]:.1e} / ref {e[1]:.1e}" for n, e in zip(names, errs)))
    return got, ref, gold


if __name__ == "__main__":
    cases = [
        (2, 4, 2, 128, 128, 1024, 256, False, False),
        (2, 4, 2, 128, 128, 1024, 256, False, True),
        (2, 4, 2, 128, 128, 1024, 256, True, True),
        (1, 2, 1, 256, 512, 1536, 512, True, False),
        (2, 8, 4, 256, 256, 16384, 4096, False, True),
    ]
    for c in cases:
        check(*c)
