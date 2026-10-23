import os, sys, time
S = "/tmp/claude-243000396/-data-home-johnson-xllm/d3839bfb-7816-4668-9945-7c337f0c3f9e/scratchpad/handoff/scripts/awm"
sys.path.insert(0, os.path.join(S, "xext")); sys.path.insert(0, "/home/johnson/xllm")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
import torch, torch.distributed as dist, types
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
preset, bsz, seq_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dist.init_process_group("nccl", world_size=1, rank=0); torch.cuda.set_device(0)
from xllm.distributed import initialize_model_parallel
initialize_model_parallel(1, 1, 1800)
import xllm.modules.fused_ops
from xllm.models.gekko import Gekko
from xllm.configuration import ConfStore
class Tok: bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, 32000, 32000
for sb in ("swift", "cute"):
    cfg = ConfStore[preset]; cfg.vocab_size = Tok.vocab_size; cfg.fused_block = True; cfg.awm_backend = "cute"; cfg.timenorm_backend = "cute"; cfg.sca_backend = sb
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
        tnw = [p.grad.float().norm().item() for n, p in model.named_parameters() if "timenorm" in n and p.grad is not None][:2]
        print(f"{preset} sca={sb} step {s}: loss {loss.float().mean().item():.6f} gnorm {gn:.6f} timenorm grads {tnw} {dt:.0f} ms", flush=True)
        model.zero_grad(set_to_none=True)
    print(f"{preset} sca={sb}: peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True); torch.cuda.reset_peak_memory_stats()
    del model; torch.cuda.empty_cache()
dist.destroy_process_group()
