import importlib.util, sys, torch
_s = importlib.util.spec_from_file_location("cute_sca", "/home/johnson/xllm/xllm/modules/fused_ops/cute/sliding_chunk_attention.py")
sca = importlib.util.module_from_spec(_s); _s.loader.exec_module(sca)
torch.manual_seed(0)
B, H, K, S, V, L, C = [int(x) for x in sys.argv[1].split(",")[:7]]; seg_p = float(sys.argv[1].split(",")[7])
dev = "cuda"
q = torch.randn(B, L, H, S, device=dev).to(torch.bfloat16); k = torch.randn(B, L, K, S, device=dev).to(torch.bfloat16)
v = torch.randn(B, L, K, V, device=dev).to(torch.bfloat16)
bos = torch.rand(B, L, device=dev) < seg_p; bos[:, 0] = True; seg = torch.cumsum(bos.long(), 1) - 1
kmin = sca.segment_starts(seg, B, L, dev); scale = 1.0 / S ** 0.5
for _ in range(2): sca.sca_fwd(q, k, v, C, scale, kmin)
torch.cuda.synchronize()
