"""Repeated / interleaved calls: workspace parity reuse, races, tickets. Compare every call to a reference."""
import sys, torch
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from timestep_decay_norm import timestep_decay_norm_fwd
torch.manual_seed(0)
cases = []
for (B, L, H, G, pad) in [(4, 4100, 2048, 32, 0.02), (2, 9000, 1024, 16, 0.0), (1, 300, 512, 8, 0.05)]:
    dt = torch.bfloat16
    x = torch.randn(B, L, H, device="cuda").to(dt)
    bos = torch.rand(B, L, device="cuda") < 0.002
    padm = torch.rand(B, L, device="cuda") < pad if pad > 0 else None
    pc = torch.randint(0, 50, (B,), device="cuda")
    pm = (torch.randn(B, G, device="cuda") * 0.1).to(dt); pv = (torch.rand(B, G, device="cuda") * 0.1).to(dt)
    w = torch.randn(H, device="cuda").to(dt); b = torch.randn(H, device="cuda").to(dt)
    args = (x, bos, pc, pm, pv, w, b, padm, G, 0.999, 0.9999, 1e-5)
    ref = [t.clone() for t in timestep_decay_norm_fwd(*args)]
    cases.append((args, ref))
bad = 0
for it in range(30):
    for ci, (args, ref) in enumerate(cases):
        out = timestep_decay_norm_fwd(*args)
        for name, o, r in zip(["y", "count", "mean", "var", "cm", "cr", "sc"], out, ref):
            d = (o.double() - r.double()).abs().max().item()
            if d > 1e-3 * max(1.0, r.double().abs().max().item()):
                bad += 1
                print(f"iter {it} case {ci} {name} diff {d}")
torch.cuda.synchronize()
print("stress:", "OK" if bad == 0 else f"{bad} mismatches")
