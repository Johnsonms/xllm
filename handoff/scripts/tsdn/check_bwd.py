import sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("tt", "/home/johnson/xllm/tests/normalization/test_cute_timestep_decay_norm.py")
tt = importlib.util.module_from_spec(spec); spec.loader.exec_module(tt)
tsdn = tt.tsdn
CASES = [  # B, L, H, G, dtype, bos, pad, cfg
    (2, 128, 256, 4, torch.float32, 0.0, 0.0, None),
    (3, 300, 256, 4, torch.float32, 0.02, 0.0, None),
    (2, 517, 512, 8, torch.float32, 0.02, 0.1, None),
    (2, 1000, 256, 32, torch.bfloat16, 0.01, 0.05, None),
    (1, 9000, 1024, 16, torch.bfloat16, 0.0005, 0.0, None),
    (2, 4100, 2048, 32, torch.bfloat16, 0.001, 0.01, None),
]
for B, L, H, G, dt, br, pr, cfg in CASES:
    x, bos, pc, pm, pv, w, b, pad = tt.make_inputs(B, L, H, G, dt, br, pr)
    y, cnt, M, V, cm, cr, sc = tsdn.timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, pad, G, tt.BETA1, tt.BETA2, tt.EPS)
    g = torch.Generator(device="cuda").manual_seed(1)
    dy = torch.randn(B, L, H, device="cuda", generator=g).to(dt)
    mg = torch.randn(B, G, device="cuda", generator=g).to(dt)
    vg = torch.randn(B, G, device="cuda", generator=g).to(dt)
    gx, gpm, gpv, gw, gb = tsdn.timestep_decay_norm_bwd(dy, mg, vg, x, bos, cm, cr, sc, w, pad, G, tt.BETA1, tt.BETA2, cfg)
    leaves = [t.double().requires_grad_() for t in (x, pm, pv, w, b)]
    ry, rc, rM, rV, *_ = tt.reference_fwd(leaves[0], bos, pc, leaves[1], leaves[2], leaves[3], leaves[4], G, pad)
    ((ry * dy.double()).sum() + (rM * mg.double()).sum() + (rV * vg.double()).sum()).backward()
    errs = {n: tt.rel_err(o, l.grad) for n, o, l in zip(["dx", "dpm", "dpv", "dw", "db"], [gx, gpm, gpv, gw, gb], leaves)}
    print(f"B{B} L{L} H{H} G{G} {str(dt)[6:]:8} " + " ".join(f"{k}={v:.1e}" for k, v in errs.items()), flush=True)
