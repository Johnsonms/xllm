"""AWM forward as three phases (prototype in PyTorch, any dtype; fp64 for checking):

  1. per update chunk j:  lse_j[s] = logsumexp_i k[i, s]  (masked positions excluded)
                          P_j = exp(k_j - lse_j)^T  (S x C)      A_j = P_j @ qk_j  (S x S)   B_j = P_j @ v_j  (S x V)
  2. per (b, k):          M <- (1 - r) * M + r * (B_j - A_j @ M)   (r = exp(lse_j - logaddexp(lse_j, lnt)))
  3. out_c = q_c @ M_c   (masked rows zeroed)

The update is linear in M:  kk (v - qk M) = kk v - (kk qk) M, so the large reductions over the chunk (A_j, B_j)
do not depend on the recurrence and run in parallel; the recurrence only multiplies S x S by S x V.
"""
import torch
from einops import rearrange


def chunk_masks(segment_idx, prev_segment_idx, prev_segment_count, nc, C, has_prev):
    """key_mask [B, nc, C] (chunks 0..nc-1), prev_key_mask [B, C], memory_mask [B, nc], out_mask [B, nc, C]
    exactly as in xllm.modules.fused_ops.adaptive_working_memory."""
    B = segment_idx.shape[0]
    seg = segment_idx.view(B, nc, C)
    key_mask = seg != seg[:, :, -1:]
    memory_mask = torch.zeros(B, nc, dtype=torch.bool, device=seg.device)
    out_mask = torch.zeros(B, nc, C, dtype=torch.bool, device=seg.device)
    memory_mask[:, 2:] = seg[:, :-2, -1] != seg[:, 1:-1, -1]
    out_mask[:, 2:] = seg[:, 2:] != seg[:, :-2, -1:]
    prev_key_mask = None
    if has_prev:
        prev_key_mask = prev_segment_idx != prev_segment_idx[:, -1:]
        memory_mask[:, 0] = prev_segment_count != prev_segment_idx[:, -1]
        memory_mask[:, 1] = prev_segment_idx[:, -1] != seg[:, 0, -1]
        out_mask[:, 0] = seg[:, 0] != prev_segment_count.view(B, 1)
        out_mask[:, 1] = seg[:, 1] != prev_segment_idx[:, -1:]
    return key_mask, prev_key_mask, memory_mask, out_mask


def phase1(k, qk, v, key_mask):
    """k, qk: [B, K, n, C, S], v: [B, K, n, C, V], key_mask [B, n, C] or None -> lse [B, K, n, S], A, B."""
    kf = k.double() if k.dtype != torch.float64 else k
    if key_mask is not None:
        kf = kf.masked_fill(key_mask[:, None, :, :, None], float("-inf"))
    lse = torch.logsumexp(kf, dim=3)  # B K n S
    P = torch.exp(kf - lse.unsqueeze(3))  # B K n C S
    A = torch.einsum("bkncs,bknct->bknst", P, qk.to(P.dtype))
    Bm = torch.einsum("bkncs,bkncv->bknsv", P, v.to(P.dtype))
    return lse, A, Bm


def awm_fwd_phased(query, qkey, key, value, C, memory, log_norm_term, prev_qkey, prev_key, prev_value,
                   segment_idx=None, prev_segment_count=None):
    """Same arguments / layouts as _adaptive_working_memory_fwd (B x H x L x S, ...). Computes in fp64."""
    Bsz, H, L, S = query.shape
    K, V = value.shape[1], value.shape[3]
    nc = L // C
    n_rep = H // K
    dev = query.device
    has_prev = prev_key is not None
    prev_seg = None
    if segment_idx is not None and has_prev:
        prev_seg, segment_idx = segment_idx[:, :C], segment_idx[:, C:]
    masks = chunk_masks(segment_idx, prev_seg, prev_segment_count, nc, C, has_prev) if segment_idx is not None else None

    # update chunks: data of chunk c-1 for the update at chunk c (prev chunk for c = 0)
    k_ch = rearrange(key, "b k (n c) s -> b k n c s", c=C)
    qk_ch = rearrange(qkey, "b k (n c) s -> b k n c s", c=C)
    v_ch = rearrange(value, "b k (n c) v -> b k n c v", c=C)
    lse, A, Bm = phase1(k_ch[:, :, :-1], qk_ch[:, :, :-1], v_ch[:, :, :-1],
                        masks[0][:, :-1] if masks is not None else None)
    if has_prev:
        lse0, A0, B0 = phase1(prev_key.unsqueeze(2), prev_qkey.unsqueeze(2), prev_value.unsqueeze(2),
                              masks[1].unsqueeze(1) if masks is not None else None)
        lse, A, Bm = (torch.cat([x0, x], dim=2) for x0, x in ((lse0, lse), (A0, A), (B0, Bm)))
    else:  # no update at chunk 0
        pad = lambda x: torch.cat([torch.zeros_like(x[:, :, :1]), x], dim=2)
        lse = torch.cat([torch.full_like(lse[:, :, :1], float("-inf")), lse], dim=2)
        A, Bm = pad(A), pad(Bm)

    # phase 2: recurrence
    if memory is None:
        M = torch.zeros(Bsz, K, S, V, dtype=torch.float64, device=dev)
        lnt = torch.full((Bsz, K, S), float("-inf"), dtype=torch.float64, device=dev)
    else:
        M = memory.double().clone()
        lnt = log_norm_term.double().clone()
    Ms = []
    out_mask = torch.zeros(Bsz, nc, C, dtype=torch.bool, device=dev) if masks is None else masks[3].clone()
    for c in range(nc):
        Ms.append(M)
        out_mask[:, c] |= torch.isinf(lnt[:, 0, 0]).view(Bsz, 1)
        if c == 0 and not has_prev:
            continue
        if masks is not None:
            mm = masks[2][:, c].view(Bsz, 1, 1)
            lnt = lnt.masked_fill(mm, float("-inf"))
            M = M.masked_fill(mm.unsqueeze(-1), 0.0)
        new_lnt = torch.logaddexp(lse[:, :, c], lnt)
        r = torch.exp(lse[:, :, c] - new_lnt).unsqueeze(-1)
        M = (1 - r) * M + r * (Bm[:, :, c] - A[:, :, c] @ M)
        lnt = new_lnt
    if memory is None:
        out_mask[:, 0] = True

    # phase 3: outputs
    Mseq = torch.stack(Ms, dim=2)  # B K n S V
    Mseq = Mseq.repeat_interleave(n_rep, dim=1)  # B H n S V
    qd = rearrange(query.double(), "b h (n c) s -> b h n c s", c=C)
    out = torch.einsum("bhncs,bhnsv->bhncv", qd, Mseq)
    out = out.masked_fill(out_mask[:, None, :, :, None], 0.0)
    return rearrange(out, "b h n c v -> b h (n c) v"), rearrange(out_mask, "b n c -> b (n c)"), M, lnt
