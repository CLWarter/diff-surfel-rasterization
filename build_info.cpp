#include <torch/extension.h>
#include <string>

#include "cuda_rasterizer/lighting_config.cuh"
#include "build_info.h"

#define STR1(x) #x
#define STR(x) STR1(x)

static std::string lighting_build_info_string() {
    std::string s;
    s += "LIGHT_ENABLE_FWD=" STR(LIGHT_ENABLE_FWD) "\n";
    s += "LIGHT_ENABLE_BWD=" STR(LIGHT_ENABLE_BWD) "\n";
    s += "LIGHT_MODE_NO_LIGHTING=" STR(LIGHT_MODE_NO_LIGHTING) "\n";
    s += "LIGHT_MODE_LAMBERT_ONLY=" STR(LIGHT_MODE_LAMBERT_ONLY) "\n";
    s += "LIGHT_MODE_PHONG_ONLY=" STR(LIGHT_MODE_PHONG_ONLY) "\n";
    s += "LIGHT_MODE_LAMBERT_PHONG=" STR(LIGHT_MODE_LAMBERT_PHONG) "\n";
    s += "LIGHT_AMBIENT_MODE=" STR(LIGHT_AMBIENT_MODE) "\n";
    s += "LIGHT_AMBIENT_FIXED=" STR(LIGHT_AMBIENT_FIXED) "\n";
    s += "LIGHT_LAMBERT_ABS=" STR(LIGHT_LAMBERT_ABS) "\n";
    s += "LIGHT_USE_SPOT=" STR(LIGHT_USE_SPOT) "\n";
    s += "LIGHT_SPOT_INNER_DEG=" STR(LIGHT_SPOT_INNER_DEG) "\n";
    s += "LIGHT_SPOT_OUTER_DEG=" STR(LIGHT_SPOT_OUTER_DEG) "\n";
    s += "LIGHT_SPOT_EXP=" STR(LIGHT_SPOT_EXP) "\n";
    s += "LIGHT_PHONG_KS=" STR(LIGHT_PHONG_KS) "\n";
    s += "LIGHT_PHONG_SHININESS=" STR(LIGHT_PHONG_SHININESS) "\n";
    s += "LIGHT_SPEC_GATING=" STR(LIGHT_SPEC_GATING) "\n";
    s += "LIGHT_ENERGY_COMP=" STR(LIGHT_ENERGY_COMP) "\n";

    return s;
}

torch::Tensor get_lighting_build_info() {
    auto s = lighting_build_info_string();
    auto out = torch::empty({(long long)s.size()},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(out.data_ptr(), s.data(), s.size());
    return out;
}
