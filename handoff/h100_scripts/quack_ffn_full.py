import sys,types,torch
sys.path.insert(0,'/home/johnson/quack')
sys.modules.setdefault('flash_attn_2_cuda', types.ModuleType('x'))
from torch.profiler import profile, ProfilerActivity
from quack.gemm_interface import gemm_gated, gemm_dgated, gemm
from xllm.modules.fused_ops.cute.swiglu import swiglu_fwd, swiglu_bwd
import torch.nn.functional as F
def t(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in p.key_averages())/10
for name,M,D,E in [('0.9B',32768,1536,4096),('4B',8192,2560,9216),('7.1B',8192,4096,8192)]:
    g=torch.Generator(device='cuda').manual_seed(0)
    x=torch.randn(M,D,device='cuda',generator=g).bfloat16(); res=torch.randn(M,D,device='cuda',generator=g).bfloat16()
    w1=(torch.randn(E,D,device='cuda',generator=g)*0.02).bfloat16(); w3=(torch.randn(E,D,device='cuda',generator=g)*0.02).bfloat16()
    w2=(torch.randn(D,E,device='cuda',generator=g)*0.02).bfloat16(); dy=torch.randn(M,D,device='cuda',generator=g).bfloat16()
    def ours():
        h1=F.linear(x,w1); h3=F.linear(x,w3); h=swiglu_fwd(h1,h3); out=torch.addmm(res,h,w2.t())
        hg=torch.mm(dy,w2); h,dh1,dh3=swiglu_bwd(h1,h3,hg)
        dx=dh1.matmul(w1); dx=torch.addmm(dx,dh3,w3,out=dx)
        g1=torch.mm(dh1.t(),x); g3=torch.mm(dh3.t(),x); g2=torch.mm(dy.t(),h)
        return out,dx,g1,g3,g2
    def qk():
        W=torch.cat([w1,w3]); pre,h=gemm_gated(x,W.t(),activation='swiglu',concat_layout=("B",)); out=torch.addmm(res,h,w2.t())
        W=torch.cat([w1,w3]); dpre,h=gemm_dgated(dy,w2,pre,activation='swiglu',dynamic_scheduler=True)
        dx=gemm(dpre,W,dynamic_scheduler=True,concat_layout=("B",))
        gW=gemm(dpre.t(),x,dynamic_scheduler=True,concat_layout=("out",)); g2=torch.mm(dy.t(),h)
        return out,dx,gW[:E],gW[E:],g2
    a=ours(); b=qk()
    errs=[((u.float()-v.float()).norm()/v.float().norm()).item() for u,v in zip(b,a)]
    to,tq=[],[]
    for _ in range(5): to.append(t(ours)); tq.append(t(qk))
    print(f"{name}: ours min {min(to):6.0f} med {sorted(to)[2]:6.0f} us  quack min {min(tq):6.0f} med {sorted(tq)[2]:6.0f} us  rel diff out,dx,g1,g3,g2 {['%.1e'%e for e in errs]}")
