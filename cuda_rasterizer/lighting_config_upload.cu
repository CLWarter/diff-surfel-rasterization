#include <cuda_runtime.h>
#include <torch/extension.h>

#include "lighting_config.cuh"
#include "lighting_config_upload.cuh"

__device__ __constant__ LightingConfig d_lighting_cfg;

void upload_lighting_config_to_cuda(const LightingConfig& cfg)
{
    cudaError_t err = cudaMemcpyToSymbol(
        d_lighting_cfg, &cfg, sizeof(LightingConfig), 0, cudaMemcpyHostToDevice);
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol(d_lighting_cfg) failed: ", cudaGetErrorString(err));
}

// Layout indices (sync with python)
enum {
    I_ENABLE_FWD = 0,
    I_ENABLE_BWD = 1,
    I_LIGHT_MODE = 2,
    I_AMBIENT_MODE = 3,
    I_AMBIENT_FIXED = 4,
    I_LAMBERT_MODE = 5,
    I_PHONG_KS_MODE = 6,
    I_PHONG_SHINY_MODE = 7,
    I_PHONG_KS = 8,
    I_PHONG_SHININESS = 9,
    I_SPEC_GATING = 10,
    I_ENERGY_COMP = 11,
    I_USE_SPOT = 12,
    I_SPOT_INNER = 13,
    I_SPOT_OUTER = 14,
    I_SPOT_EXP = 15,
    CFG_LEN = 16
};

void SetLightingConfigCUDA(torch::Tensor cfg_tensor_cpu)
{
    TORCH_CHECK(cfg_tensor_cpu.device().is_cpu(), "lighting cfg must be a CPU tensor");
    TORCH_CHECK(cfg_tensor_cpu.is_contiguous(), "lighting cfg must be contiguous");
    TORCH_CHECK(cfg_tensor_cpu.numel() == CFG_LEN, "lighting cfg must have length ", CFG_LEN);
    TORCH_CHECK(cfg_tensor_cpu.scalar_type() == torch::kFloat32,
                "lighting cfg tensor must be float32");

    const float* p = cfg_tensor_cpu.data_ptr<float>();

    LightingConfig cfg;
    cfg.enable_fwd       = (int32_t)p[I_ENABLE_FWD];
    cfg.enable_bwd       = (int32_t)p[I_ENABLE_BWD];
    cfg.light_mode       = (int32_t)p[I_LIGHT_MODE];
    cfg.ambient_mode     = (int32_t)p[I_AMBIENT_MODE];
    cfg.ambient_fixed    =        p[I_AMBIENT_FIXED];
    cfg.lambert_mode     = (int32_t)p[I_LAMBERT_MODE];
    cfg.phong_ks_mode    = (int32_t)p[I_PHONG_KS_MODE];
    cfg.phong_shiny_mode = (int32_t)p[I_PHONG_SHINY_MODE];
    cfg.phong_ks         =        p[I_PHONG_KS];
    cfg.phong_shininess  =        p[I_PHONG_SHININESS];
    cfg.spec_gating      = (int32_t)p[I_SPEC_GATING];
    cfg.energy_comp      = (int32_t)p[I_ENERGY_COMP];
    cfg.use_spot         = (int32_t)p[I_USE_SPOT];
    cfg.spot_inner_deg   =        p[I_SPOT_INNER];
    cfg.spot_outer_deg   =        p[I_SPOT_OUTER];
    cfg.spot_exp         =        p[I_SPOT_EXP];

    upload_lighting_config_to_cuda(cfg);
}
