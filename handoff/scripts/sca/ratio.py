import importlib.util, sys, torch
_s = importlib.util.spec_from_file_location("cute_sca", "/home/johnson/xllm/xllm/modules/fused_ops/cute/sliding_chunk_attention.py")
sca = importlib.util.module_from_spec(_s); _s.loader.exec_module(sca)
for a in sys.argv[1:]:
    torch.manual_seed(0)
    B, H, K, S, V, L, C = [int(x) for x in a.split(",")[:7]]; seg_p = float(a.split(",")[7])
    dev = "cuda"
    q = torch.randn(B, L, H, S, device=dev).to(torch.bfloat16); k = torch.randn(B, L, K, S, device=dev).to(torch.bfloat16)
    v = torch.randn(B, L, K, V, device=dev).to(torch.bfloat16); do = torch.randn(B, L, H, V, device=dev).to(torch.bfloat16)
    bos = torch.rand(B, L, device=dev) < seg_p; bos[:, 0] = True; seg = torch.cumsum(bos.long(), 1) - 1
    kmin = sca.segment_starts(seg, B, L, dev); qmax = sca.query_limits(kmin)
    nc = L // C
    k0 = torch.arange(0, L, 128, device=dev)
    ck = k0 // C
    q_end = torch.minimum(qmax[:, k0 + 127].long(), torch.minimum((ck + 2) * C - 1, torch.tensor(L - 1, device=dev)))
    attended = (q_end - k0 + 1).clamp(min=0).sum()
    full = (torch.minimum((ck + 2) * C - 1, torch.tensor(L - 1, device=dev)) - k0 + 1).sum() * B
    print(f"{a}: segments {int(bos.sum())} attended/full = {attended.item() / full.item():.3f}")
