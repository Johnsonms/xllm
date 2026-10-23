import importlib.util, sys, time, torch
_s = importlib.util.spec_from_file_location("cute_sca", "/home/johnson/xllm/xllm/modules/fused_ops/cute/sliding_chunk_attention.py")
sca = importlib.util.module_from_spec(_s); _s.loader.exec_module(sca)
torch.manual_seed(0)
def run(B, H, K, S, V, L, C, seg_p, check=True, iters=20):
    dev = "cuda"
    q = torch.randn(B, L, H, S, device=dev).to(torch.bfloat16)
    k = torch.randn(B, L, K, S, device=dev).to(torch.bfloat16)
    v = torch.randn(B, L, K, V, device=dev).to(torch.bfloat16)
    do = torch.randn(B, L, H, V, device=dev).to(torch.bfloat16)
    seg = None
    if seg_p > 0:
        bos = torch.rand(B, L, device=dev) < seg_p
        bos[:, 0] = True
        seg = torch.cumsum(bos.long(), 1) - 1
    kmin = sca.segment_starts(seg, B, L, dev); qmax = sca.query_limits(kmin)
    scale = 1.0 / S ** 0.5
    y, lse2 = sca.sca_fwd(q, k, v, C, scale, kmin)
    dq, dk, dv = sca.sca_bwd(q, k, v, y, lse2, do, C, scale, kmin, qmax)
    torch.cuda.synchronize()
    if check:
        qf, kf, vf = [t.float().requires_grad_(True) for t in (q, k, v)]
        yr, _ = sca.sca_reference(qf, kf, vf, C, scale, seg)
        yr.backward(do.float())
        for name, g, r in (("dq", dq, qf.grad), ("dk", dk, kf.grad), ("dv", dv, vf.grad)):
            err = (g.float() - r).abs().max().item()
            print(f"B{B} H{H} K{K} S{S} V{V} L{L} C{C} seg={seg_p}: max|d{name}| {err:.3e} (ref max {r.abs().max().item():.2f})")
    for _ in range(3): sca.sca_bwd(q, k, v, y, lse2, do, C, scale, kmin, qmax)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): sca.sca_bwd(q, k, v, y, lse2, do, C, scale, kmin, qmax)
    torch.cuda.synchronize(); dt = (time.time() - t0) / iters * 1e6
    print(f"   bwd {dt:.0f} us")
for a in sys.argv[1:]:
    parts = a.split(","); run(*[int(x) for x in parts[:7]], float(parts[7]), check=(len(parts) < 9 or parts[8] != "nocheck"))
