"""xattn SCA bwd (bos route) at the 0.9B shape: GQA (KV 4) vs K/V expanded to 8 heads (MHA) + group sums."""
import torch
from torch.profiler import profile, ProfilerActivity
from xattn.ops import flash_sca_fwd, flash_sca_bwd
B, H, KV, S, V, L, C = 2, 8, 4, 256, 256, 16384, 4096
g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16(); k = torch.randn(B, L, KV, S, device="cuda", generator=g).bfloat16()
v = torch.randn(B, L, KV, V, device="cuda", generator=g).bfloat16(); dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
bos = torch.rand(B, L, device="cuda", generator=g) < 1 / 3000; bos[:, 0] = True
def gt(fn, it=5):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(it): fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in p.key_averages()) / it
sc = S ** -0.5
y, _, lse = flash_sca_fwd(q, k, v, C, sc, None, None, bos_mask=bos)
gq = flash_sca_bwd(dy, q, k, v, y, lse, C, sc, None, None, bos_mask=bos)[:3]
t_gqa = gt(lambda: flash_sca_bwd(dy, q, k, v, y, lse, C, sc, None, None, bos_mask=bos))
ke, ve = k.repeat_interleave(H // KV, 2), v.repeat_interleave(H // KV, 2)
y2, _, lse2 = flash_sca_fwd(q, ke, ve, C, sc, None, None, bos_mask=bos)
def mha_bwd():
    dq, dk, dv = flash_sca_bwd(dy, q, ke, ve, y2, lse2, C, sc, None, None, bos_mask=bos)[:3]
    return dq, dk.view(B, L, KV, H // KV, S).sum(3), dv.view(B, L, KV, H // KV, V).sum(3)
gm = mha_bwd()
t_mha = gt(mha_bwd)
print(f"GQA bwd {t_gqa:.0f} us | MHA-expanded bwd + group sums {t_mha:.0f} us; diffs",
      [f"{(a.float() - b.float()).abs().max().item():.2e}" for a, b in zip(gq, gm)])
