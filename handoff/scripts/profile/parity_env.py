"""Loss / grad-norm parity of a model between values of an env var read at call time.
usage: parity_env.py <preset> <B> <L> <ENV> <v1,v2,...> [XEXT path]"""
import os, sys, time
xext = sys.argv[6] if len(sys.argv) > 6 else "/home/johnson/scratch/xext"
sys.path.insert(0, xext); sys.path.insert(0, "/home/johnson/xllm")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29600")
import torch, torch.distributed as dist, types
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
preset, bsz, seq_len, env, vals = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
dist.init_process_group("nccl", world_size=1, rank=0); torch.cuda.set_device(0)
from xllm.distributed import initialize_model_parallel
initialize_model_parallel(1, 1, 1800)
import xllm.modules.fused_ops
from xllm.models.gekko import Gekko
from xllm.configuration import ConfStore
class Tok: bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, 32000, 32000
torch.backends.cuda.matmul.allow_tf32 = True
for v in vals:
    os.environ[env] = v
    cfg = ConfStore[preset]; cfg.vocab_size = Tok.vocab_size; cfg.fused_block = True; cfg.awm_backend = "cute"; cfg.timenorm_backend = "cute"; cfg.sca_backend = "cute"
    torch.manual_seed(0); model = Gekko(cfg, Tok()).cuda().to(torch.bfloat16); model.train()
    g = torch.Generator(device="cuda").manual_seed(0)
    for s in range(3):
        tokens = torch.randint(3, Tok.vocab_size, (bsz, seq_len + 1), device="cuda", generator=g)
        tokens[torch.rand(bsz, seq_len + 1, device="cuda", generator=g) < 1 / 3000] = Tok.eos_id
        x, y = tokens[:, :-1], tokens[:, 1:]
        torch.cuda.synchronize(); t0 = time.time()
        loss = model(x, True, targets=y); loss = loss[0] if isinstance(loss, tuple) else loss
        loss.float().mean().backward(); torch.cuda.synchronize(); dt = (time.time() - t0) * 1000
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
        sel = {n.split(".")[-2] + "." + n.split(".")[-1]: p.grad.float().norm().item() for n, p in model.named_parameters() if p.grad is not None and ("layers.1." in n) and any(k in n for k in ("qnorm", "wq", "wk", "ffn_norm", "timenorm_w"))}
        print(f"{preset} {env}={v} step {s}: loss {loss.float().mean().item():.6f} gnorm {gn:.6f} {dt:.0f} ms  " + " ".join(f"{k}={x:.5f}" for k, x in sel.items()), flush=True)
        model.zero_grad(set_to_none=True)
    del model; torch.cuda.empty_cache()
dist.destroy_process_group()
