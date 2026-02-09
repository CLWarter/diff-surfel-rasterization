#pragma once
#include <torch/extension.h>
#include "lighting_config.cuh"

extern __device__ __constant__ LightingConfig d_lighting_cfg;

// Host callable function (pybind will expose this)
void SetLightingConfigCUDA(torch::Tensor cfg_tensor_cpu);
