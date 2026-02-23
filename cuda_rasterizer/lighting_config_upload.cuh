#pragma once
#include <torch/extension.h>
#include "lighting_config.cuh"

// Host callable function (pybind will expose this)
void SetLightingConfigCUDA(torch::Tensor cfg_tensor_cpu);
