#include <torch/extension.h>
#include <string>
#include <cstring>
#include <sstream>

#include "cuda_rasterizer/lighting_config.cuh"
#include "build_info.h"

#define STR1(x) #x
#define STR(x) STR1(x)

static std::string lighting_build_info_string() {
    std::ostringstream ss;

    // Master switches
    ss << "LIGHT_ENABLE_FWD=" << STR(LIGHT_ENABLE_FWD) << "\n";
    ss << "LIGHT_ENABLE_BWD=" << STR(LIGHT_ENABLE_BWD) << "\n";

    // Mode presets
    ss << "LIGHT_MODE_NO_LIGHTING=" << STR(LIGHT_MODE_NO_LIGHTING) << "\n";
    ss << "LIGHT_MODE_LAMBERT_ONLY=" << STR(LIGHT_MODE_LAMBERT_ONLY) << "\n";
    ss << "LIGHT_MODE_PHONG_ONLY=" << STR(LIGHT_MODE_PHONG_ONLY) << "\n";
    ss << "LIGHT_MODE_LAMBERT_PHONG=" << STR(LIGHT_MODE_LAMBERT_PHONG) << "\n";

    // Derived toggles (useful to print too)
    ss << "LIGHT_USE_LAMBERT=" << STR(LIGHT_USE_LAMBERT) << "\n";
    ss << "LIGHT_USE_PHONG=" << STR(LIGHT_USE_PHONG) << "\n";

    // Ambient
    ss << "LIGHT_AMBIENT_MODE=" << STR(LIGHT_AMBIENT_MODE) << "\n";
    ss << "LIGHT_AMBIENT_FIXED=" << STR(LIGHT_AMBIENT_FIXED) << "\n";

    // Lambert
    ss << "LIGHT_LAMBERT_ABS=" << STR(LIGHT_LAMBERT_ABS) << "\n";

    // Phong learning modes + ranges
    ss << "LIGHT_PHONG_KS_MODE=" << STR(LIGHT_PHONG_KS_MODE) << "\n";
    ss << "LIGHT_PHONG_SHININESS_MODE=" << STR(LIGHT_PHONG_SHININESS_MODE) << "\n";
    ss << "LIGHT_SHINY_MIN=" << STR(LIGHT_SHINY_MIN) << "\n";
    ss << "LIGHT_SHINY_MAX=" << STR(LIGHT_SHINY_MAX) << "\n";

    // Phong fixed fallbacks
    ss << "LIGHT_PHONG_KS=" << STR(LIGHT_PHONG_KS) << "\n";
    ss << "LIGHT_PHONG_SHININESS=" << STR(LIGHT_PHONG_SHININESS) << "\n";

    // Spec controls
    ss << "LIGHT_SPEC_GATING=" << STR(LIGHT_SPEC_GATING) << "\n";
    ss << "LIGHT_ENERGY_COMP=" << STR(LIGHT_ENERGY_COMP) << "\n";

    // Intensity / falloff
    ss << "FALLOFF_MODE=" << STR(FALLOFF_MODE) << "\n";
    ss << "FALLOFF_K=" << STR(FALLOFF_K) << "\n";
    ss << "FALLOFF_Z_GRAD_SCALE=" << STR(FALLOFF_Z_GRAD_SCALE) << "\n";
    ss << "FALLOFF_Z_GRAD_CLAMP=" << STR(FALLOFF_Z_GRAD_CLAMP) << "\n";

    // Spotlight
    ss << "LIGHT_USE_SPOT=" << STR(LIGHT_USE_SPOT) << "\n";
    ss << "LIGHT_SPOT_INNER_DEG=" << STR(LIGHT_SPOT_INNER_DEG) << "\n";
    ss << "LIGHT_SPOT_OUTER_DEG=" << STR(LIGHT_SPOT_OUTER_DEG) << "\n";
    ss << "LIGHT_SPOT_EXP=" << STR(LIGHT_SPOT_EXP) << "\n";

    // Constants
    ss << "LIGHT_PI=" << STR(LIGHT_PI) << "\n";

    return ss.str();
}

torch::Tensor get_lighting_build_info() {
    const std::string s = lighting_build_info_string();
    auto out = torch::empty({(long long)s.size()},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(out.data_ptr(), s.data(), s.size());
    return out;
}