import importlib.util, sys, time, torch
_s = importlib.util.spec_from_file_location("cute_sca", "/home/johnson/xllm/xllm/modules/fused_ops/cute/sliding_chunk_attention.py")
sca = importlib.util.module_from_spec(_s); _s.loader.exec_module(sca)
torch.manual_seed(0)
for a in sys.argv[1:]:
    B, H, K, S, V, L, C = [int(x) for x in a.split(",")[:7]]; seg_p = float(a.split(",")[7])
    dev = "cuda"
    q = torch.randn(B, L, H, S, device=dev).to(torch.bfloat16); k = torch.randn(B, L, K, S, device=dev).to(torch.bfloat16)
    v = torch.randn(B, L, K, V, device=dev).to(torch.bfloat16); do = torch.randn(B, L, H, V, device=dev).to(torch.bfloat16)
    g = torch.Generator(device=dev).manual_seed(123)
    bos = torch.rand(B, L, device=dev, generator=g) < seg_p; bos[:, 0] = True; seg = torch.cumsum(bos.long(), 1) - 1
    kmin = sca.segment_starts(seg, B, L, dev); qmax = sca.query_limits(kmin); scale = 1.0 / S ** 0.5
    y, lse2 = sca.sca_fwd(q, k, v, C, scale, kmin)
    for _ in range(3): sca.sca_bwd(q, k, v, y, lse2, do, C, scale, kmin, qmax)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): sca.sca_bwd(q, k, v, y, lse2, do, C, scale, kmin, qmax)
    torch.cuda.synchronize(); print(f"{a}: segments {int(bos.sum())}: bwd {(time.time() - t0) / 10 * 1e6:.0f} us")
