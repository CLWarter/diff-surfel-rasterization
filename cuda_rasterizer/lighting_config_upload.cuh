#pragma once
#include <torch/extension.h>

// Host callable function (pybind will expose this)
void SetLightingConfigCUDA(torch::Tensor cfg_tensor_cpu);
