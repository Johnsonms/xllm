import sys,types,torch
sys.modules.setdefault('flash_attn_2_cuda', types.ModuleType('x'))
from torch.profiler import profile, ProfilerActivity
import xllm.modules.fused_ops.attention.sca as sca
B,L,H,KV,S,V,C=1,8192,4,4,256,2048,4096
rate=float(sys.argv[1]) if len(sys.argv)>1 else 1/3000
g=torch.Generator(device='cuda').manual_seed(0)
q=torch.randn(B,L,H,S,device='cuda',generator=g).bfloat16(); k=torch.randn(B,L,KV,S,device='cuda',generator=g).bfloat16(); v=torch.randn(B,L,KV,V,device='cuda',generator=g).bfloat16(); dy=torch.randn(B,L,H,V,device='cuda',generator=g).bfloat16()
seg=torch.cumsum((torch.rand(B,L,device='cuda',generator=g)<rate).long(),1)
i=torch.arange(L,device='cuda')
mask=(i[None,:]<=i[:,None])&(i[None,:]>=((i[:,None]//C-1).clamp(min=0)*C))&(seg[0][:,None]==seg[0][None,:])
qf,kf,vf=(t.float().requires_grad_(True) for t in (q,k,v))
s=torch.einsum('blhd,bmhd->bhlm',qf,kf)*S**-0.5
p=s.masked_fill(~mask,float('-inf')).softmax(-1)
y=torch.einsum('bhlm,bmhd->blhd',p,vf); y.backward(dy.float())
ref=[y.detach(),qf.grad,kf.grad,vf.grad]; del s,p
def run():
    yy,yb,w=sca.sliding_chunk_attention_fwd(q,k,v,C,S**-0.5,None,None,None,seg,0.0,0,'swift',True)
    gr=sca.sliding_chunk_attention_bwd(dy,q,k,v,yb,w,C,S**-0.5,None,None,None,seg,0,False,'swift')
    return (yy,)+tuple(gr[:3]), w.numel()
def t(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(5): fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in pr.key_averages())/5
print('segments', seg.max().item(), 'blocks', sca._swift_seg_blocks(seg, C))
for name,sp,skip,sps in [('split1',1,False,(1,)),('split2',2,False,(2,)),('seg auto',2,True,(2,4))]:
    sca._SWIFT_QSPLIT=sp; sca._SWIFT_SEG_SKIP=skip; sca._SWIFT_QSPLITS=sps; sca._swift_seg_cache.clear()
    out,n=run()
    err=[f"{((a.float()-b).norm()/b.norm()).item():.2e}" for a,b in zip(out,ref)]
    print(f"{name:11s} fwd+bwd {t(run):6.0f} us  attn_w {n*2/2**20:5.0f} MiB  rel err y,dq,dk,dv {err}")
