"""Record the shapes the gekko fused block passes to the xllm_extension norm ops (one training step, 1 GPU)."""
import os, sys, types
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29599")
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
import torch, torch.distributed as dist
dist.init_process_group("nccl", world_size=1, rank=0); torch.cuda.set_device(0)
from xllm.distributed import initialize_model_parallel
initialize_model_parallel(1, 1, 1800)
import xllm_extension.ops as ops
seen = {}
for name in [n for n in dir(ops) if "norm" in n and ("rms" in n or "layer" in n)]:
    fn = getattr(ops, name)
    def wrap(*a, _fn=fn, _name=name):
        key = (_name, tuple(tuple(t.shape) if torch.is_tensor(t) else t for t in a[:3]))
        seen[key] = seen.get(key, 0) + 1
        return _fn(*a)
    setattr(ops, name, wrap)
import xllm.models.fused_blocks.utils as U
U.xllm_ops = ops
from xllm.models.gekko import Gekko
from xllm.configuration import ConfStore
class Tok: bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, 32000, 32000
cfg = ConfStore[sys.argv[1]]; cfg.vocab_size = 32000; cfg.fused_block = True
cfg.awm_backend = "cute"; cfg.timenorm_backend = "cute"; cfg.sca_backend = "xattn"
from xllm.models import build_model
model = build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True, reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
B, L = int(sys.argv[2]), int(sys.argv[3])
x = torch.randint(3, 32000, (B, L), device="cuda"); y = torch.randint(3, 32000, (B, L), device="cuda")
loss = model(tokens=x, targets=y, token_mask=None, multi_segments=True, deterministic=False)[0]
loss.sum().backward()
for (n, shp), c in sorted(seen.items()):
    print(f"{c:4d}x  {n}  {shp}")
