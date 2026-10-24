import importlib.util, sys, time, torch
_s = importlib.util.spec_from_file_location("cute_ce", "/home/johnson/xllm/xllm/modules/fused_ops/cute/cross_entropy.py")
ce = importlib.util.module_from_spec(_s); _s.loader.exec_module(ce)
torch.manual_seed(0)
def timeit(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for (C, V) in [(2048, 32000), (2048, 128256), (77, 1000), (5, 8)]:
    x = (torch.randn(C, V, device="cuda") * 4).to(torch.bfloat16)
    tgt = torch.randint(0, V, (C,), device="cuda")
    ign = torch.rand(C, device="cuda") < 0.1
    tgt1 = tgt.clone(); tgt1[ign] = 0
    def ref():
        pl = x.float()
        m = pl.max(-1)[0]; pl = pl - m[:, None]
        ar = torch.arange(C, device="cuda")
        pred = pl[ar, tgt1].clone(); pred[ign] = 0.0
        e = torch.exp(pl); s = e.sum(-1)
        loss = torch.log(s) - pred; loss[ign] = 0.0
        sc = e / s[:, None]; sc[ar, tgt1] -= (1.0 - ign.float()); sc.masked_fill_(ign[:, None], 0.0)
        return loss, sc, sc.to(torch.bfloat16)
    l, s32, s16 = ref(); l2, s322, s162 = ce.ce_logits(x, tgt1, ign, True, True)
    print(f"C{C} V{V}: loss err {(l2-l).abs().max().item():.2e} (max {l.abs().max().item():.2f}) s32 err {(s322-s32).abs().max().item():.2e} s16 err {(s162.float()-s16.float()).abs().max().item():.2e} | torch {timeit(ref):.0f} us cute {timeit(lambda: ce.ce_logits(x, tgt1, ign, True, True)):.0f} us")
