"""xattn SCA fwd / bwd at the gekko-0.9B / 4B layer shapes: BOS-mask metadata (what xllm passes in training) vs
segment_idx metadata (same segments). GPU us per call and max differences. usage: python bos_vs_seg.py 0.9B|4B"""
import sys
import torch
from torch.profiler import profile, ProfilerActivity
from xattn.ops import flash_sca_fwd, flash_sca_bwd

SHAPES = {"0.9B": (2, 8, 4, 256, 256), "4B": (1, 32, 8, 128, 128)}
L, C = 16384, 4096
B, H, KV, S, V = SHAPES[sys.argv[1]]
scale = S ** -0.5


def gpu_time(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / iters


g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16()
k = torch.randn(B, L, KV, S, device="cuda", generator=g).bfloat16()
v = torch.randn(B, L, KV, V, device="cuda", generator=g).bfloat16()
dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
bos = torch.rand(B, L, device="cuda", generator=g) < 1 / 3000
bos[:, 0] = True
seg = torch.cumsum(bos.long(), 1) - 1
res = {}
for name, kw in (("bos", dict(bos_mask=bos)), ("seg", dict(segment_idx=seg))):
    y, y32, lse = flash_sca_fwd(q, k, v, C, scale, None, None, **kw)
    grads = flash_sca_bwd(dy, q, k, v, y, lse, C, scale, None, None, **kw)
    tf = gpu_time(lambda: flash_sca_fwd(q, k, v, C, scale, None, None, **kw))
    tb = gpu_time(lambda: flash_sca_bwd(dy, q, k, v, y, lse, C, scale, None, None, **kw))
    res[name] = (y, grads[:3])
    print(f"{sys.argv[1]} {name}: fwd {tf:7.0f} us, bwd {tb:7.0f} us", flush=True)
(y0, g0), (y1, g1) = res["bos"], res["seg"]
print("max |y diff|", (y0.float() - y1.float()).abs().max().item(),
      "grads", [(a.float() - b.float()).abs().max().item() for a, b in zip(g0, g1)])
