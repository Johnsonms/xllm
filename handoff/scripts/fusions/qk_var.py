import importlib.util, sys, time, types, torch
_s = importlib.util.spec_from_file_location("cute_qk", "/home/johnson/xllm/xllm/modules/fused_ops/cute/qk_prep.py")
qk = importlib.util.module_from_spec(_s); _s.loader.exec_module(qk)
def timeit(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
B, L, G, S, R = 2, 16384, 8, 256, 128
x = torch.randn(B, L, G * S, device="cuda").to(torch.bfloat16)
w = (torch.randn(G * S, device="cuda") * 0.1).to(torch.bfloat16)
f = 1.0 / (500000.0 ** (torch.arange(0, R, 2, device="cuda").float() / R))
fc = torch.polar(torch.ones(L, R // 2, device="cuda"), torch.outer(torch.arange(L, device="cuda").float(), f))
dy = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16); dp = torch.randn_like(dy); ex = torch.randn_like(dy)
for aff in (True, False):
    for sm in (True, False):
        for e in (True, False):
            t = timeit(lambda: qk.qk_prep_bwd(dy, dp if sm else None, ex if e else None, x, G, w if aff else None, fc, R, 1e-6))
            print(f"affine={aff} sm={sm} extra={e}: {t:.0f} us")
for aff in (True, False):
    print(f"fwd affine={aff}: {timeit(lambda: qk.qk_prep_fwd(x, G, w if aff else None, fc, R, 1e-6, want_pre=False, want_sm=True)):.0f} us")
