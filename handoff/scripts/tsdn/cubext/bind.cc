#include <torch/extension.h>
#include "ops/timestep_decay_norm_cub.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { xllm::ops::DefineTimestepDecayNormCubOp(m); }
