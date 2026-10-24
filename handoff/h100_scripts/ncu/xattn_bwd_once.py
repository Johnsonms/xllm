"""xattn SCA backward of the gekko-0.9B layer shape once (for ncu): EOS segments (bos mask) or none."""
import sys
import torch
from xattn.ops import flash_sca_fwd, flash_sca_bwd
B, H, KV, S, V, L, C = 2, 8, 4, 256, 256, 16384, 4096
g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16(); k = torch.randn(B, L, KV, S, device="cuda", generator=g).bfloat16()
v = torch.randn(B, L, KV, V, device="cuda", generator=g).bfloat16(); dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
bos = torch.rand(B, L, device="cuda", generator=g) < (1 / 3000 if sys.argv[1] == "eos" else 0)
bos[:, 0] = True
y, _, lse = flash_sca_fwd(q, k, v, C, S ** -0.5, None, None, bos_mask=bos)
for _ in range(2):
    flash_sca_bwd(dy, q, k, v, y, lse, C, S ** -0.5, None, None, bos_mask=bos)
torch.cuda.synchronize()
