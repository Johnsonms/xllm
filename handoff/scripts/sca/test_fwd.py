import importlib.util, sys, time, torch
_s = importlib.util.spec_from_file_location("cute_sca", "/home/johnson/xllm/xllm/modules/fused_ops/cute/sliding_chunk_attention.py")
sca = importlib.util.module_from_spec(_s); _s.loader.exec_module(sca)
torch.manual_seed(0)
def run(B, H, K, S, V, L, C, seg_p, check=True, iters=20):
    dev = "cuda"
    q = torch.randn(B, L, H, S, device=dev).to(torch.bfloat16)
    k = torch.randn(B, L, K, S, device=dev).to(torch.bfloat16)
    v = torch.randn(B, L, K, V, device=dev).to(torch.bfloat16)
    seg = None
    if seg_p > 0:
        bos = torch.rand(B, L, device=dev) < seg_p
        bos[:, 0] = True
        seg = torch.cumsum(bos.long(), 1) - 1
    kmin = sca.segment_starts(seg, B, L, dev)
    scale = 1.0 / S ** 0.5
    y, lse2 = sca.sca_fwd(q, k, v, C, scale, kmin)
    torch.cuda.synchronize()
    if check:
        yr, lr = sca.sca_reference(q, k, v, C, scale, seg)
        err = (y.float() - yr).abs().max().item(); lerr = (lse2 - lr).abs().max().item()
        print(f"B{B} H{H} K{K} S{S} V{V} L{L} C{C} seg={seg_p}: max|dy| {err:.3e} (ref max {yr.abs().max().item():.2f}) max|dlse2| {lerr:.3e}")
    for _ in range(3): sca.sca_fwd(q, k, v, C, scale, kmin)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): sca.sca_fwd(q, k, v, C, scale, kmin)
    torch.cuda.synchronize(); dt = (time.time() - t0) / iters * 1e6
    print(f"   {dt:.0f} us")
for a in sys.argv[1:]:
    parts = a.split(","); run(*[int(x) for x in parts[:7]], float(parts[7]), check=(len(parts) < 9 or parts[8] != "nocheck"))
