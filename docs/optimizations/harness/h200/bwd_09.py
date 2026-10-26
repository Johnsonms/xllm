"""xattn SCA D = V = 256 backward per layer at the gekko-0.9B shape (B2 L16384 H8 KV4, chunk 4096), the H100
hand-off's table (EOS 1/3000: 4025 us total / 3705 main with pair reuse; 1/1000: 2586 / 2258; no EOS 7214)."""
import torch
from torch.profiler import profile, ProfilerActivity
from xattn.ops import flash_sca_fwd, flash_sca_bwd
B, L, H, KV, S, C = 2, 16384, 8, 4, 256, 4096
torch.manual_seed(0)
q = torch.randn(B, L, H, S, device="cuda").bfloat16(); k = torch.randn(B, L, KV, S, device="cuda").bfloat16()
v = torch.randn(B, L, KV, S, device="cuda").bfloat16(); dy = torch.randn(B, L, H, S, device="cuda").bfloat16()
for p in (1 / 3000, 1 / 1000, None):
    kw = {}
    if p is not None:
        bos = torch.rand(B, L, device="cuda") < p; bos[:, 0] = True; kw = dict(bos_mask=bos)
    y, _, lse = flash_sca_fwd(q, k, v, C, S ** -0.5, None, None, **kw)
    run = lambda: flash_sca_bwd(dy, q, k, v, y, lse, C, S ** -0.5, None, None, **kw)
    for _ in range(3): run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(5): run()
        torch.cuda.synchronize()
    ka = pr.key_averages()
    tot = sum(e.self_device_time_total for e in ka) / 5
    main = sum(e.self_device_time_total for e in ka if "AttentionBwdSm90" in e.key) / 5
    print(f"EOS {p}: bwd total {tot:.0f} us, main kernel {main:.0f} us")
