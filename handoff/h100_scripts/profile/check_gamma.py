import torch, xllm_extension.ops as ops
torch.manual_seed(0)
B, L = 2, 16384
for D, G in [(2048, 8), (1024, 4), (256, 2)]:
    x = torch.randn(B, L, D, device="cuda").bfloat16(); dy = torch.randn_like(x); w = torch.randn(D, device="cuda").bfloat16()
    y, rstd = ops.group_rms_norm_fwd_affine(x, D, G, w, 1e-6)
    dx, dg = ops.group_rms_norm_bwd_affine(dy, x, D, G, rstd, w, False)
    xh = x.double().view(B, L, G, D // G) * rstd.double().view(B, L, G, 1)
    ref = (dy.double().view(B, L, G, D // G) * xh).sum((0, 1)).view(D)
    print(f"rms D{D} G{G}: dgamma rel err {((dg.double() - ref).abs().max() / ref.abs().max()).item():.2e}")
for D, G in [(1536, 2), (512, 1)]:
    x = torch.randn(B, L, D, device="cuda").bfloat16(); dy = torch.randn_like(x); w = torch.randn(D, device="cuda").bfloat16(); bb = torch.randn(D, device="cuda").bfloat16()
    y, mean, rstd = ops.group_layer_norm_fwd_affine(x, D, G, w, bb, 1e-6)
    dx, dg, db = ops.group_layer_norm_bwd_affine(dy, x, D, G, mean, rstd, w, bb, False)
    xh = (x.double().view(B, L, G, D // G) - mean.double().view(B, L, G, 1)) * rstd.double().view(B, L, G, 1)
    rg = (dy.double().view(B, L, G, D // G) * xh).sum((0, 1)).view(D); rb = dy.double().sum((0, 1))
    print(f"layer D{D} G{G}: dgamma rel err {((dg.double() - rg).abs().max() / rg.abs().max()).item():.2e}, dbeta {((db.double() - rb).abs().max() / rb.abs().max()).item():.2e}")
