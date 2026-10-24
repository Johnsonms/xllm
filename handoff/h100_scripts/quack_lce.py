import sys,types,torch
sys.modules.setdefault('flash_attn_2_cuda', types.ModuleType('x'))
from torch.profiler import profile, ProfilerActivity
from quack.linear_cross_entropy import scaled_exp_linear_cross_entropy, chunked_linear_cross_entropy, scaled_exp_lce_supported
from xllm.models.fused_blocks.cross_entropy import _fused_linear_cross_entropy
def t(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5): fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in p.key_averages())/5
for name,M,D,V in [('0.9B',32768,1536,32000),('7.1B',8192,4096,32000),('0.9B V128k',32768,1536,128256)]:
    g=torch.Generator(device='cuda').manual_seed(0)
    x=torch.randn(M,D,device='cuda',generator=g).bfloat16(); w=(torch.randn(V,D,device='cuda',generator=g)*0.02).bfloat16()
    tgt=torch.randint(0,V,(M,),device='cuda',generator=g)
    CH=int(sys.argv[1]); print(name,'supported', scaled_exp_lce_supported(x,w,CH,'sum'))
    xs=x.view(1,M,D)
    ours=lambda: _fused_linear_cross_entropy(xs, w, tgt.view(1,M), None, None, -100, 0.0, True, True, False)
    def qk():
        xx=x.detach().requires_grad_(True); ww=w.detach().requires_grad_(True)
        l=scaled_exp_linear_cross_entropy(xx,ww,tgt,reduction='sum',chunk_size=CH) if scaled_exp_lce_supported(x,w,CH,'sum') else chunked_linear_cross_entropy(xx,ww,tgt,reduction='sum',chunk_size=4096)
        l.backward(); return l,xx.grad,ww.grad
    lo,gi,gw=ours(); lq,gx,gwq=qk()
    print(f"  loss ours {lo.sum().item():.3f} quack {lq.item():.3f}; dx rel {((gx.float()-gi.float()).norm()/gi.float().norm()).item():.1e} dw rel {((gwq.float()-gw.float()).norm()/gw.float().norm()).item():.1e}")
    print(f"  ours {t(ours):7.0f} us   quack {t(qk):7.0f} us")
