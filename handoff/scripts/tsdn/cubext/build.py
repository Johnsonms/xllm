import os, torch
from torch.utils.cpp_extension import load
S = "/home/johnson/xllm/xllm/csrc"
here = os.path.dirname(os.path.abspath(__file__))
ext = load(
    name="tsdn_cub",
    sources=[os.path.join(here, "bind.cc"), f"{S}/ops/timestep_decay_norm_cub.cc", f"{S}/ops/timestep_decay_norm_cub_kernel.cu"],
    extra_include_paths=[S],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-gencode=arch=compute_100,code=sm_100"],
    extra_ldflags=[f"-L{here}/libs"],
    build_directory=os.path.join(here, "build"),
    verbose=False,
)
print("built", ext)
