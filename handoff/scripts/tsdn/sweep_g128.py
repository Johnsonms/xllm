import sys, torch
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from timestep_decay_norm import timestep_decay_norm_fwd
from bench_cub import bench
dt = torch.bfloat16
B, L, H, G = 8, 8192, 2048, 128
x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
pc = torch.zeros(B, dtype=torch.int64, device="cuda"); pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
sol = (2 * x.numel() * 2 + B * G * L * 8 + B * L * 4) / 6.4 / 1e6
for cfg in [(4, 256, 256, 1, 1, 1), (4, 128, 256, 2, 1, 1), (4, 64, 256, 3, 1, 1), (2, 128, 256, 3, 1, 1), (2, 256, 256, 2, 1, 1), (1, 256, 256, 3, 1, 1), (8, 64, 256, 2, 1, 1)]:
    t = bench(lambda: timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5, cfg))
    print(f"cfg={cfg}: {t:7.1f} us  {100*sol/t:5.1f}% SOL", flush=True)
