import importlib.util, sys, torch
from collections import defaultdict
_s = importlib.util.spec_from_file_location("cute_sca", "/home/johnson/xllm/xllm/modules/fused_ops/cute/sliding_chunk_attention.py")
sca = importlib.util.module_from_spec(_s); _s.loader.exec_module(sca)
torch.manual_seed(0)
B, H, K, S, V, L, C = [int(x) for x in sys.argv[1].split(",")[:7]]; seg_p = float(sys.argv[1].split(",")[7])
dev = "cuda"
q = torch.randn(B, L, H, S, device=dev).to(torch.bfloat16); k = torch.randn(B, L, K, S, device=dev).to(torch.bfloat16)
v = torch.randn(B, L, K, V, device=dev).to(torch.bfloat16); do = torch.randn(B, L, H, V, device=dev).to(torch.bfloat16)
bos = torch.rand(B, L, device=dev) < seg_p; bos[:, 0] = True; seg = torch.cumsum(bos.long(), 1) - 1
kmin = sca.segment_starts(seg, B, L, dev); qmax = sca.query_limits(kmin); scale = 1.0 / S ** 0.5
y, lse2 = sca.sca_fwd(q, k, v, C, scale, kmin)
for _ in range(2): sca.sca_bwd(q, k, v, y, lse2, do, C, scale, kmin, qmax)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3): sca.sca_bwd(q, k, v, y, lse2, do, C, scale, kmin, qmax)
    torch.cuda.synchronize()
tot = defaultdict(float)
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA: tot[e.name[:90]] += e.device_time / 3
print(f"{sys.argv[1]}: total {sum(tot.values()):.0f} us")
for n, t in sorted(tot.items(), key=lambda x: -x[1])[:12]: print(f"  {t:8.0f} us  {n}")
