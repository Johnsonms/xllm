"""Float64 reference for group timestep decay norm (forward via a time loop, backward via autograd)."""
import torch


def tsdn_ref(x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, num_groups,
             padding_mask=None, beta1=0.999, beta2=0.9999, eps=1e-5):
    B, L, H = x.shape
    G = num_groups
    xd = x.double().view(B, L, G, H // G)
    mt_all = xd.mean(-1)                          # B x L x G
    vt_all = xd.var(-1, correction=0)             # B x L x G
    bos = bos_mask if bos_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=x.device)
    pad = padding_mask if padding_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=x.device)

    count = prev_count.clone()
    M = prev_mean.double()
    V = prev_var.double()
    cms, crs = [], []
    for t in range(L):
        live = (~pad[:, t]).view(B, 1)
        reset = (bos[:, t] & ~pad[:, t]).view(B, 1)
        decay1 = torch.where(reset, 0.0, beta1)
        decay2 = torch.where(reset, 0.0, beta2)
        M = torch.where(live, decay1 * M + (1 - beta1) * mt_all[:, t], M)
        V = torch.where(live, decay2 * V + (1 - beta2) * vt_all[:, t], V)
        count = torch.where(live.view(B), torch.where(reset.view(B), 1, count + 1), count)
        c = count.double().view(B, 1)
        den1 = 1 - beta1 ** c
        den2 = 1 - beta2 ** c
        ok = (den1 > 0) & (den2 > 0)
        cm = torch.where(ok, M / den1.clamp_min(1e-300), 0.0)
        cr = torch.where(ok, torch.rsqrt(V / den2.clamp_min(1e-300) + eps), eps ** -0.5)
        cms.append(cm)
        crs.append(cr)
    cummean = torch.stack(cms, dim=2)  # B x G x L
    cumrstd = torch.stack(crs, dim=2)
    xn = (xd - cummean.transpose(1, 2).unsqueeze(-1)) * cumrstd.transpose(1, 2).unsqueeze(-1)
    y = xn.reshape(B, L, H) * gamma.double() + beta.double()
    y = y.masked_fill(pad.unsqueeze(-1), 0.0)
    return y, count, M, V, cummean, cumrstd
