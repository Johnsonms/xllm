"""AWM backward in the phase decomposition (PyTorch, fp64), checked against autograd through the FP64 reference.

Forward (per (b, k); update index u uses chunk j(u) = u - 1 + first... see awm_proto):
  P_j = softmax_t(k_j masked)^T (S x C),  lse_j = logsumexp_t k_j
  A_j = P_j qk_j (S x S),  B_j = P_j v_j (S x V)
  lnt' = logaddexp(lse_j, lnt),  r = exp(lse_j - lnt')
  M' = (1 - r) M + r (B_j - A_j M)                         (M <- 0, lnt <- -inf on a memory reset first)
  out_c = q_c M_c  (masked rows 0)

Backward phases (mirror of the forward):
  3'. dq_c = dOut_c M_c^T;   dMo_c = sum_h q_c^T dOut_c                (parallel over chunks)
  2'. reverse recurrence over updates:  with G = dL/dM' and g = dL/dlnt'
        D = B - A M - M  (so M' = M + r D);  dr = sum_v G * D
        dB = r G;  dA = -(r G) M^T;  dM = (1 - r) G - A^T (r G) + dMo
        r = exp(lse - lnt'): dlse += dr r;  dlnt' -= dr r
        lnt' = logaddexp(lse, lnt): dlse += g' w1, dlnt += g' w2 with w1 = exp(lse - lnt'), w2 = exp(lnt - lnt')
  1'. per update chunk (parallel): dqk = P^T dA (C x S), dv = P^T dB (C x V)
        dP = dA qk^T + dB v^T (S x C); dk[t, s] = P[s, t] (dP[s, t] - sum_t' P[s, t'] dP[s, t']) + P[s, t] dlse[s]
"""
import torch
from einops import rearrange


def awm_fwd_saved(q, qk, k, v, C, memory, lnt, prev_qk, prev_k, prev_v, segment_idx, prev_segment_count):
    """Forward (no prev / memory for brevity of the prototype: memory=None, no prev; segments supported)
    returning the tensors the phased backward needs."""
    B, H, L, S = q.shape
    K, V = v.shape[1], v.shape[3]
    nc, n_rep = L // C, H // K
    assert memory is None and prev_k is None
    kch = rearrange(k, "b k (n c) s -> b k n c s", c=C)[:, :, :-1]
    qkch = rearrange(qk, "b k (n c) s -> b k n c s", c=C)[:, :, :-1]
    vch = rearrange(v, "b k (n c) v -> b k n c v", c=C)[:, :, :-1]
    if segment_idx is not None:
        seg = segment_idx.view(B, nc, C)
        kmask = (seg != seg[:, :, -1:])[:, :-1]
        memory_mask = torch.zeros(B, nc, dtype=torch.bool, device=q.device)
        memory_mask[:, 2:] = seg[:, :-2, -1] != seg[:, 1:-1, -1]
        out_mask = torch.zeros(B, nc, C, dtype=torch.bool, device=q.device)
        out_mask[:, 2:] = seg[:, 2:] != seg[:, :-2, -1:]
        kf = kch.masked_fill(kmask[:, None, :, :, None], float("-inf"))
    else:
        memory_mask = torch.zeros(B, nc, dtype=torch.bool, device=q.device)
        out_mask = torch.zeros(B, nc, C, dtype=torch.bool, device=q.device)
        kf = kch
    lse = torch.logsumexp(kf, dim=3)  # B K nu S
    P = torch.exp(kf - lse.unsqueeze(3))  # B K nu C S
    A = torch.einsum("bknts,bkntu->bknsu", P, qkch)
    Bm = torch.einsum("bknts,bkntv->bknsv", P, vch)
    M = torch.zeros(B, K, S, V, dtype=q.dtype, device=q.device)
    lt = torch.full((B, K, S), float("-inf"), dtype=q.dtype, device=q.device)
    Ms, lts, rs, Mpre = [], [], [], []
    for c in range(nc):
        Ms.append(M)
        out_mask[:, c] |= torch.isinf(lt[:, 0, 0]).view(B, 1)
        if c == 0:
            continue
        u = c - 1
        mm = memory_mask[:, c].view(B, 1, 1)
        lt = lt.masked_fill(mm, float("-inf"))
        M = M.masked_fill(mm.unsqueeze(-1), 0.0)
        Mpre.append(M)
        new = torch.logaddexp(lse[:, :, u], lt)
        r = torch.exp(lse[:, :, u] - new)
        lts.append((lt, new))
        rs.append(r)
        M = (1 - r.unsqueeze(-1)) * M + r.unsqueeze(-1) * (Bm[:, :, u] - A[:, :, u] @ M)
        lt = new
    out_mask[:, 0] = True
    Mseq = torch.stack(Ms, 2)
    out = torch.einsum("bhncs,bhnsv->bhncv", rearrange(q, "b h (n c) s -> b h n c s", c=C),
                       Mseq.repeat_interleave(n_rep, 1))
    out = out.masked_fill(out_mask[:, None, :, :, None], 0.0)
    saved = dict(P=P, A=A, Bm=Bm, lse=lse, Mseq=Mseq, Mpre=Mpre, lts=lts, rs=rs, memory_mask=memory_mask,
                 out_mask=out_mask, kch=kch, qkch=qkch, vch=vch, kmask=kmask if segment_idx is not None else None)
    return rearrange(out, "b h n c v -> b h (n c) v"), M, lt, saved


def awm_bwd_phased(dout, dM_fin, dlnt_fin, q, qk, k, v, C, saved):
    B, H, L, S = q.shape
    K, V = v.shape[1], v.shape[3]
    nc, n_rep = L // C, H // K
    P, A, Bm, lse, Mseq = saved["P"], saved["A"], saved["Bm"], saved["lse"], saved["Mseq"]
    dout = rearrange(dout, "b h (n c) v -> b h n c v", c=C).masked_fill(saved["out_mask"][:, None, :, :, None], 0.0)
    qc = rearrange(q, "b h (n c) s -> b h n c s", c=C)
    # 3'
    dq = torch.einsum("bhncv,bhnsv->bhncs", dout, Mseq.repeat_interleave(n_rep, 1))
    dMo = torch.einsum("bhncs,bhncv->bhnsv", qc, dout).view(B, K, n_rep, nc, S, V).sum(2)
    # 2'
    G = dM_fin.clone()
    g = dlnt_fin.clone()
    nu = nc - 1
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(Bm)
    dlse = torch.zeros_like(lse)
    for u in reversed(range(nu)):
        c = u + 1
        M = saved["Mpre"][u]
        r = saved["rs"][u].unsqueeze(-1)
        lt_old, lt_new = saved["lts"][u]
        D = Bm[:, :, u] - A[:, :, u] @ M - M
        dr = (G * D).sum(-1)  # B K S
        rG = r * G
        dB[:, :, u] = rG
        dA[:, :, u] = -rG @ M.transpose(-1, -2)
        dMp = (1 - r) * G - A[:, :, u].transpose(-1, -2) @ rG
        # r = exp(lse - lnt'), lnt' = logaddexp(lse, lnt)
        rr = r.squeeze(-1)
        g_new = g - dr * rr
        dl = dr * rr
        w1 = torch.exp(lse[:, :, u] - lt_new)
        w2 = torch.exp(lt_old - lt_new)
        dl = dl + g_new * w1
        g = g_new * torch.nan_to_num(w2, nan=0.0)
        dlse[:, :, u] = dl
        # memory reset before the update: gradients do not flow to the state / normalizer before it
        mm = saved["memory_mask"][:, c].view(B, 1, 1)
        dMp = dMp.masked_fill(mm.unsqueeze(-1), 0.0)
        g = g.masked_fill(mm, 0.0)
        G = dMp + dMo[:, :, c]  # dL/dM_c: through the update and through out_c
    # dM for chunk 0's state (initial memory) is G after the loop (unused: memory=None)
    # 1'
    qkch, vch = saved["qkch"], saved["vch"]
    dqk_ch = torch.einsum("bknts,bknsu->bkntu", P, dA)
    dv_ch = torch.einsum("bknts,bknsv->bkntv", P, dB)
    dP = torch.einsum("bknsu,bkntu->bknts", dA, qkch) + torch.einsum("bknsv,bkntv->bknts", dB, vch)
    dk_ch = P * (dP - (P * dP).sum(3, keepdim=True)) + P * dlse.unsqueeze(3)
    dqk = torch.zeros_like(qk)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dqk[:, :, :L - C] = rearrange(dqk_ch, "b k n c s -> b k (n c) s")
    dk[:, :, :L - C] = rearrange(dk_ch, "b k n c s -> b k (n c) s")
    dv[:, :, :L - C] = rearrange(dv_ch, "b k n c v -> b k (n c) v")
    return rearrange(dq, "b h n c s -> b h (n c) s"), dqk, dk, dv


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/johnson/xllm/tests")
    from test_cute_adaptive_working_memory import reference_fwd
    torch.manual_seed(0)
    for (B, H, K, S, V, L, C, seg_rate) in [(2, 4, 2, 16, 24, 256, 64, 0.0), (2, 4, 2, 16, 24, 384, 64, 0.6),
                                            (1, 2, 1, 32, 48, 512, 128, 1.0)]:
        dev = "cuda"
        q = torch.softmax(torch.randn(B, H, L, S, device=dev, dtype=torch.float64), -1).requires_grad_()
        kr = torch.randn(B, K, L, S, device=dev, dtype=torch.float64) * 2
        k = kr.clone().requires_grad_()
        qk = torch.softmax(kr, -1).requires_grad_()
        v = torch.randn(B, K, L, V, device=dev, dtype=torch.float64).requires_grad_()
        seg = (torch.cumsum((torch.rand(B, L, device=dev) < seg_rate / C).long(), 1) + 3) if seg_rate > 0 else None
        out, mask, M, lt = reference_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
        dout = torch.randn_like(out)
        dM = torch.randn_like(M)
        dl = torch.randn_like(lt)
        loss = (out * dout).sum() + (M * dM).sum() + (lt * dl).sum()
        loss.backward()
        with torch.no_grad():
            o2, M2, lt2, saved = awm_fwd_saved(q, qk, k, v, C, None, None, None, None, None, seg, None)
            dq, dqk, dk, dv = awm_bwd_phased(dout, dM, dl, q, qk, k, v, C, saved)
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
        print(f"B{B} H{H} K{K} S{S} V{V} L{L} C{C} seg={seg_rate}: fwd {rel(o2, out):.1e} M {rel(M2, M):.1e} | "
              f"dq {rel(dq, q.grad):.1e} dqk {rel(dqk, qk.grad):.1e} dk {rel(dk, k.grad):.1e} dv {rel(dv, v.grad):.1e}")
