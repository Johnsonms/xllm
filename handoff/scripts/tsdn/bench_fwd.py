import sys, torch
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
sys.path.insert(0, "cubext/build")
import tsdn_cub
from timestep_decay_norm import timestep_decay_norm_fwd
from bench_cub import SHAPES, bench

PEAK = 6.40  # TB/s, measured copy bandwidth
cfg = None if len(sys.argv) < 2 else tuple(int(v) for v in sys.argv[1].split(","))
dt = torch.bfloat16
print(f"cfg={cfg}")
print(f"{'B':>3} {'L':>6} {'H':>5} {'G':>4} | {'cub us':>8} {'cute us':>8} {'SOL us':>7} {'%SOL':>6} {'speedup':>8}")
for B, L, H, G in SHAPES:
    x = torch.randn(B, L, H, dtype=dt, device="cuda")
    bos = torch.rand(B, L, device="cuda") < 0.001
    pc = torch.zeros(B, dtype=torch.int64, device="cuda")
    pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
    w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
    tc = bench(lambda: tsdn_cub.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5))
    tn = bench(lambda: timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5, cfg))
    nbytes = x.numel() * x.element_size()
    sol = (2 * nbytes + B * G * L * 8 + B * L * 4) / PEAK / 1e6
    print(f"{B:>3} {L:>6} {H:>5} {G:>4} | {tc:8.1f} {tn:8.1f} {sol:7.1f} {100*sol/tn:6.1f} {tc/tn:7.1f}x")
