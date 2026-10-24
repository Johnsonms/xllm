import sys, time, torch
sys.path.insert(0, sys.argv[1])
import xllm_extension
ops = xllm_extension.ops
torch.manual_seed(0)
def timeit(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for (rows, D, G) in [(32768, 2048, 8), (8192, 4096, 32), (16384, 1024, 4), (16384, 4096, 4), (32768, 1536, 2)]:
    x = torch.randn(2, rows // 2, D, device="cuda").to(torch.bfloat16); g = torch.randn(2, rows // 2, D, device="cuda").to(torch.bfloat16)
    w = (torch.rand(D, device="cuda") * 0.5 + 0.75).to(torch.bfloat16); b = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    # rms affine
    y, rstd = ops.group_rms_norm_fwd_affine(x, D, G, w, 1e-6)
    xg, wg = ops.group_rms_norm_bwd_affine(g, x, D, G, rstd, w, False)
    xn = x.float().view(rows, G, D // G) * rstd.view(rows, G, 1)
    ref = (g.float().view(rows, G, D // G) * xn).sum(0).reshape(-1)
    t = timeit(lambda: ops.group_rms_norm_bwd_affine(g, x, D, G, rstd, w, False))
    print(f"rms  rows{rows} D{D} G{G}: wgrad max err {(wg.float() - ref).abs().max().item():.3e} (ref max {ref.abs().max().item():.1f}) bwd_affine {t:.0f} us")
    # layer norm affine
    y, mean, rstd = ops.group_layer_norm_fwd_affine(x, D, G, w, b, 1e-5)
    xg, wg, bg = ops.group_layer_norm_bwd_affine(g, x, D, G, mean, rstd, w, b, False)
    xn = (x.float().view(rows, G, D // G) - mean.view(rows, G, 1)) * rstd.view(rows, G, 1)
    ref = (g.float().view(rows, G, D // G) * xn).sum(0).reshape(-1); refb = g.float().reshape(-1, D).sum(0)
    t = timeit(lambda: ops.group_layer_norm_bwd_affine(g, x, D, G, mean, rstd, w, b, False))
    print(f"ln   rows{rows} D{D} G{G}: wgrad err {(wg.float() - ref).abs().max().item():.3e} bgrad err {(bg.float() - refb).abs().max().item():.3e} (ref max {ref.abs().max().item():.1f}) bwd_affine {t:.0f} us")
