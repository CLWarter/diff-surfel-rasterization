#pragma once
#include <stdint.h>

// ------------------ MODES ------------------------

// ------------------ Light modes ------------------
// Allow testing forward-only / backward-only lighting
enum LightMode : int32_t {
  NO_LIGHTING = 0,
  LAMBERT_ONLY = 1,
  PHONG_ONLY = 2,
  LAMBERT_PHONG = 3
};

// ------------------ Ambient modes ------------------
// 0 = off
// 1 = fixed constant
// 2 = learned, sigmoid(ambients[0])
enum AmbientMode : int32_t {
  AMBIENT_OFF = 0,
  AMBIENT_FIXED = 1,
  AMBIENT_LEARN = 2
};

// ------------------ Lambert options ------------------
// 1 = clamp(max(ndotl,0))
// 0 = abs(ndotl)
enum LambertMode : int32_t {
  LAMBERT_COS = 0,
  LAMBERT_ABS = 1
};

// ------------------ Phong parameters ------------------
enum PhongKSMode : int32_t {
  KS_HARDCODED = 0,
  KS_LEARN = 1
};

enum PhongShinyMode: int32_t {
  SHINY_HARDCODED = 0,
  SHINY_LEARN = 1
};

// ------------------ Specular gating ------------------
// 0 = none
// 1 = backface gate only: spec = 0 if ndotl <= 0
// 2 = scale by lambert
enum SpecGatingMode : int32_t {
  SPEC_GATING_OFF = 0,
  SPEC_GATING_BACKFACE = 1,
  SPEC_GATING_LAMBERTSCALE = 2
};

// ------------------ Energy conservation (compensation) ------------------
// 0 = none
// 1 = diffuse *= (1-ks)
enum EnergyCompMode: int32_t {
  ENERGY_COMP_OFF = 0,
  ENERGY_COMP_ON = 1
};

// ------------------ Spotlight falloff ------------------
// 0 = off
// 1 = on
enum UseSpotMode : int32_t {
  LIGHT_SPOT_OFF = 0,
  LIGHT_SPOT_ON = 1
};

// ------------------ STRUCT ------------------

struct LightingConfig {
  // master
  int32_t enable_fwd = 1;
  int32_t enable_bwd = 1;

  // light mode preset
  int32_t light_mode = LightMode::LAMBERT_PHONG;

  // ambient
  int32_t ambient_mode = AmbientMode::AMBIENT_LEARN;
  float ambient_fixed = 0.02f;

  // lambert
  int32_t lambert_mode = LambertMode::LAMBERT_COS;

  // phong
  int32_t phong_ks_mode = PhongKSMode::KS_LEARN;
  int32_t phong_shiny_mode = PhongShinyMode::SHINY_HARDCODED;
  float phong_ks = 0.1f;
  float phong_shininess = 8.0f;

  // spec gating
  int32_t spec_gating = SpecGatingMode::SPEC_GATING_LAMBERTSCALE;

  // energy compensation
  int32_t energy_comp = EnergyCompMode::ENERGY_COMP_ON;

  // spotlight
  int32_t use_spot = UseSpotMode::LIGHT_SPOT_ON;
  float spot_inner_deg = 15.0f;
  float spot_outer_deg = 35.0f;
  float spot_exp = 1.5f;
};

// Light constants
#ifndef LIGHT_PI
#define LIGHT_PI 3.14159265358979323846f
#endif

__device__ __forceinline__ const LightingConfig& get_lighting_cfg() {
    return d_lighting_cfg;
}

#ifdef __CUDACC__
__device__ __forceinline__ bool light_use_lambert(const LightingConfig& c) {
    return (c.light_mode == LAMBERT_ONLY) || (c.light_mode == LAMBERT_PHONG);
}
__device__ __forceinline__ bool light_use_phong(const LightingConfig& c) {
    return (c.light_mode == PHONG_ONLY) || (c.light_mode == LAMBERT_PHONG);
}
#endif