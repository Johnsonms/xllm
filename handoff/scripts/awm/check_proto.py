"""Check the phased AWM forward prototype against xllm's _adaptive_working_memory_fwd (fp32 inputs)."""
import os
import sys
import types

S_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S_DIR, "xext"))
sys.path.insert(0, "/home/johnson/xllm")
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
import torch  # noqa: E402
import xllm.modules.fused_ops  # noqa: E402,F401
ref_mod = sys.modules["xllm.modules.fused_ops.adaptive_working_memory"]
from awm_proto import awm_fwd_phased  # noqa: E402


SEG_RATE = float(os.environ.get("SEG_RATE", "0.4"))


def make(B, H, K, S, V, L, C, with_memory, with_seg, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device="cuda", generator=g)
    q = torch.softmax(rn(B, H, L, S) * 2, -1).to(dtype)
    k = (rn(B, K, L, S) * 2).to(dtype)
    qk = torch.softmax(k.float(), -1).to(dtype)
    v = rn(B, K, L, V).to(dtype)
    memory = lnt = pk = pqk = pv = None
    if with_memory:
        memory = (rn(B, K, S, V) * 0.1).to(dtype)
        lnt = rn(B, K, S) + 8.0
        pk = (rn(B, K, C, S) * 2).to(dtype)
        pqk = torch.softmax(pk.float(), -1).to(dtype)
        pv = rn(B, K, C, V).to(dtype)
    seg = cnt = None
    if with_seg:
        T = L + (C if with_memory else 0)
        bos = torch.rand(B, T, device="cuda", generator=g) < SEG_RATE / C
        seg = torch.cumsum(bos.long(), dim=1) + 3
        if with_memory:
            cnt = seg[:, 0] - (torch.rand(B, device="cuda", generator=g) < 0.5).long()
    return q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


if __name__ == "__main__":
  for (B, H, K, S, V, L, C, mem, segf) in [
      (2, 4, 2, 32, 48, 256, 64, False, False),
      (2, 4, 2, 32, 48, 384, 64, False, True),
      (2, 4, 2, 32, 48, 384, 64, True, True),
      (3, 2, 2, 16, 64, 512, 64, True, False),
      (1, 4, 4, 64, 128, 1024, 128, True, True),
  ]:
      args = make(B, H, K, S, V, L, C, mem, segf)
      q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
      ro, rm, rM, rl = ref_mod._adaptive_working_memory_fwd(q, qk, k, v, C, memory, lnt, pqk, pk, pv, seg, cnt)
      po, pm, pM, pl = awm_fwd_phased(q, qk, k, v, C, memory, lnt, pqk, pk, pv, seg, cnt)
      print(f"B{B} H{H} K{K} S{S} V{V} L{L} C{C} mem={mem} seg={segf}: out {rel(po, ro):.1e} "
            f"mask_eq={torch.equal(pm, rm)} M {rel(pM, rM):.1e} lnt {rel(pl, rl):.1e} "
            f"(masked frac {rm.float().mean().item():.2f})")
