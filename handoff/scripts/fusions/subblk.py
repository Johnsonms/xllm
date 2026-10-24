import torch, sys, time
dev = "cuda"
nc, B, H, C, V, S = 4, 1, 4, 4096, 2048, 256
pm = torch.randn(nc, B, H, C, 2 * C, device=dev).to(torch.bfloat16)
doc = torch.randn(nc, B, H, C, V, device=dev).to(torch.bfloat16)
vc = torch.randn(nc, B, H, C, V, device=dev).to(torch.bfloat16)
cur = (slice(None),) * 4 + (slice(C, None),)
def timeit(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
# dense dV cur
def dv_dense():
    return torch.bmm(pm[cur].reshape(-1, C, C).transpose(1, 2), doc.view(-1, C, V))
# sub-blocked: for row block i: dV[0:(i+1)c] += P[i, 0:(i+1)c]^T dO[i]
def dv_sub(n):
    c = C // n
    out = torch.empty(nc, B, H, C, V, dtype=torch.bfloat16, device=dev)
    i = n - 1
    torch.bmm(pm[:, :, :, i*c:(i+1)*c, C:C+(i+1)*c].reshape(-1, c, (i+1)*c).transpose(1, 2), doc[:, :, :, i*c:(i+1)*c].reshape(-1, c, V), out=out.view(-1, C, V))
    for i in range(n - 1):
        out[:, :, :, :(i+1)*c].reshape(-1, (i+1)*c, V).baddbmm_(pm[:, :, :, i*c:(i+1)*c, C:C+(i+1)*c].reshape(-1, c, (i+1)*c).transpose(1, 2), doc[:, :, :, i*c:(i+1)*c].reshape(-1, c, V))
    return out
ref = dv_dense()
# zero the upper triangle of P cur for a fair compare
mask = torch.ones(C, C, device=dev, dtype=torch.bool).tril()
pm[cur] *= mask
ref = dv_dense()
for n in (2, 4, 8):
    o = dv_sub(n); print("dV n", n, "err", (o.view(-1, C, V).float() - ref.float()).abs().max().item(), "us", f"{timeit(lambda: dv_sub(n)):.0f}")
print("dV dense us", f"{timeit(dv_dense):.0f}")
# dP cur: dP[i, 0:(i+1)c] = dO[i] V[0:(i+1)c]^T  fp32 out
def dp_dense():
    return torch.bmm(doc.view(-1, C, V), vc.view(-1, C, V).transpose(1, 2), out_dtype=torch.float32)
def dp_sub(n):
    c = C // n
    out = torch.empty(nc, B, H, C, C, dtype=torch.float32, device=dev)
    for i in range(n):
        torch.bmm(doc[:, :, :, i*c:(i+1)*c].reshape(-1, c, V), vc[:, :, :, :(i+1)*c].reshape(-1, (i+1)*c, V).transpose(1, 2), out_dtype=torch.float32, out=out[:, :, :, i*c:(i+1)*c, :(i+1)*c].reshape(-1, c, (i+1)*c))
    return out
ref = dp_dense()
for n in (2, 4, 8):
    o = dp_sub(n); print("dP n", n, "err", (o.view(-1, C, C).tril(0).float() - ref.tril(0).float()).abs().max().item(), "us", f"{timeit(lambda: dp_sub(n)):.0f}")
print("dP dense us", f"{timeit(dp_dense):.0f}")
