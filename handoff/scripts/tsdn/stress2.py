import sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("tt", "/home/johnson/xllm/tests/normalization/test_cute_timestep_decay_norm.py")
tt = importlib.util.module_from_spec(spec); spec.loader.exec_module(tt)
tsdn = tt.tsdn
names = ["y", "count", "mean", "var", "cm", "cr", "sc"]
cases = []
for B, L, H, G, pr in [(4, 4100, 2048, 32, 0.02), (2, 9000, 1024, 16, 0.0), (1, 300, 512, 8, 0.05)]:
    x, bos, pc, pm, pv, w, b, pad = tt.make_inputs(B, L, H, G, torch.bfloat16, 0.002, pr, seed=B * L)
    call = (x, bos, pc, pm, pv, w, b, pad, G, tt.BETA1, tt.BETA2, tt.EPS)
    ref = tt.reference_fwd(x, bos, pc, pm, pv, w, b, G, pad)
    cases.append((call, ref))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
bad = 0
for it in range(n):
    for ci, (call, ref) in enumerate(cases):
        out = tsdn.timestep_decay_norm_fwd(*call)
        for k, (o, r) in enumerate(zip(out, ref)):
            r = r.double()
            d = (o.double() - r).abs()
            if d.max().item() > 2e-2 * max(1.0, r.abs().max().item()):
                bad += 1
                idx = (d == d.max()).nonzero()[0].tolist()
                print(f"iter {it} case {ci} {names[k]} maxdiff {d.max().item():.3g} at {idx} (#bad elems {(d > 2e-2).sum().item()})", flush=True)
torch.cuda.synchronize()
print("mismatches:", bad)
