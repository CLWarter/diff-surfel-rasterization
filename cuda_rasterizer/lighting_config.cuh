#pragma once

// ------------------ Master switches (compile-time) ------------------
// Allow testing forward-only / backward-only lighting
#ifndef LIGHT_ENABLE_FWD
#define LIGHT_ENABLE_FWD 1
#endif

#ifndef LIGHT_ENABLE_BWD
#define LIGHT_ENABLE_BWD 1
#endif

// ------------------ Lighting mode presets ------------------
// Pick just one
#ifndef LIGHT_MODE_NO_LIGHTING
#define LIGHT_MODE_NO_LIGHTING 0
#endif
#ifndef LIGHT_MODE_LAMBERT_ONLY
#define LIGHT_MODE_LAMBERT_ONLY 0
#endif
#ifndef LIGHT_MODE_PHONG_ONLY
#define LIGHT_MODE_PHONG_ONLY 0
#endif
#ifndef LIGHT_MODE_LAMBERT_PHONG
#define LIGHT_MODE_LAMBERT_PHONG 1   // default, what should be best
#endif

#if (LIGHT_MODE_NO_LIGHTING + LIGHT_MODE_LAMBERT_ONLY + LIGHT_MODE_PHONG_ONLY + LIGHT_MODE_LAMBERT_PHONG) != 1
#error "Select exactly one lighting mode: NO_LIGHTING / LAMBERT_ONLY / PHONG_ONLY / LAMBERT_PHONG"
#endif

// Derived feature toggles from mode
#if LIGHT_MODE_NO_LIGHTING
  #define LIGHT_USE_LAMBERT 0
  #define LIGHT_USE_PHONG   0
#elif LIGHT_MODE_LAMBERT_ONLY
  #define LIGHT_USE_LAMBERT 1
  #define LIGHT_USE_PHONG   0
#elif LIGHT_MODE_PHONG_ONLY
  #define LIGHT_USE_LAMBERT 0
  #define LIGHT_USE_PHONG   1
#else
  #define LIGHT_USE_LAMBERT 1
  #define LIGHT_USE_PHONG   1
#endif

// ------------------ Ambient modes ------------------
// 0 = off
// 1 = fixed constant
// 2 = learned with sigmoid(ambients[0])
#ifndef LIGHT_AMBIENT_MODE
#define LIGHT_AMBIENT_MODE 2
#endif

#ifndef LIGHT_AMBIENT_FIXED
#define LIGHT_AMBIENT_FIXED 0.02f
#endif

// ------------------ Lambert options ------------------
// 0 = clamp (max(ndotl,0))
//1 = abs(ndotl), with max being default
#ifndef LIGHT_LAMBERT_ABS
#define LIGHT_LAMBERT_ABS 0
#endif

// ------------------ Phong parameters ------------------
#ifndef LIGHT_PHONG_KS_MODE
#define LIGHT_PHONG_KS_MODE 1   // 1 to learn ks, 0 hard-coded
#endif

// 0 = fixed LIGHT_PHONG_SHININESS define
// 1 = learned per-scene shininess (scalar in [SHINY_MIN, SHINY_MAX])
#ifndef LIGHT_PHONG_SHININESS_MODE
#define LIGHT_PHONG_SHININESS_MODE 1
#endif

#define LIGHT_SHINY_MIN  2.0f
#define LIGHT_SHINY_MAX  128.0f

#ifndef LIGHT_PHONG_KS
#define LIGHT_PHONG_KS 0.10f
#endif

#ifndef LIGHT_PHONG_SHININESS
#define LIGHT_PHONG_SHININESS 16.0f
#endif

// --------------------- Intensity ---------------------
// 0 = constant intensity (uses LIGHT_INTENSITY_CONST)
// 1 = learnable intensity (uses sigmoid(intensity_raw[0]) mapped to [MIN, MAX])
#ifndef LIGHT_INTENSITY_MODE
#define LIGHT_INTENSITY_MODE 1
#endif

// If LIGHT_INTENSITY_MODE==0, this constant intensity is used.
#ifndef LIGHT_INTENSITY_CONST
#define LIGHT_INTENSITY_CONST 1.0f
#endif

// Spotlight params
#ifndef FALLOFF_MODE
#define FALLOFF_MODE 1  // 0 = none, 1 = quadratic
#endif

#ifndef FALLOFF_Z_GRAD_ENABLE
#define FALLOFF_Z_GRAD_ENABLE 1
#endif

#ifndef FALLOFF_K
#define FALLOFF_K 0.001f
#endif

#ifndef FALLOFF_Z_GRAD_SCALE
#define FALLOFF_Z_GRAD_SCALE 1.0f   // prevent depth hacks
#endif

#ifndef FALLOFF_Z_GRAD_CLAMP
#define FALLOFF_Z_GRAD_CLAMP 0.01f   // clamp added contribution to dL_dz
#endif

// --------------------- Li / Spec safety clamps ---------------------
// Hard clamp of Li = I * inv to prevent white splat spikes.
// 0 to disable the clamp.
#ifndef LIGHT_LI_CLAMP
#define LIGHT_LI_CLAMP 0
#endif


// ------------------ Specular gating ------------------
// 0 = none
// 1 = backface gate only: spec = 0 if ndotl <= 0
// 2 = scale by lambert
#ifndef LIGHT_SPEC_GATING
#define LIGHT_SPEC_GATING 2
#endif

// ------------------ Energy compensation ------------------
// 0 = none
// 1 = diffuse *= (1-ks)
#ifndef LIGHT_ENERGY_COMP
#define LIGHT_ENERGY_COMP 1
#endif

// ------------------ Spotlight falloff ------------------
// 0 = off
// 1 = on
#ifndef LIGHT_USE_SPOT
#define LIGHT_USE_SPOT 1
#endif

// Spotlight params
#ifndef LIGHT_SPOT_INNER_DEG
#define LIGHT_SPOT_INNER_DEG 20.0f
#endif

#ifndef LIGHT_SPOT_OUTER_DEG
#define LIGHT_SPOT_OUTER_DEG 50.0f
#endif

// smooth ramp
#ifndef LIGHT_SPOT_EXP
#define LIGHT_SPOT_EXP 1.0f
#endif

// Light constants
#ifndef LIGHT_PI
#define LIGHT_PI 3.14159265358979323846f
#endif


#ifndef LIGHT_DEBUG_MODE
#define LIGHT_DEBUG_MODE 0
#endif

// 0 = off
// 1 = point_cam distance to light
// 2 = falloff inv
// 3 = final Li after intensity * falloff * conf
// 4 = spotlight factor
// 5 = reliable/sane/clamped hit mask
// 6 = point_cam vs center_cam displacement
// 7 = learned raw normal
// 8 = learned ambient
// 9 = learned intensity parameter/value
// 10 = learned kspec
// 11 = learned shiny
// 12 = ndotl
// 13 = lambert
// 14 = spec_add
// 15 = point_cam.z
// 16 = alpha contribution
// 17 = w contribution
// 18 = accumulated final alpha

#ifndef LIGHT_DEBUG_SCALE
#define LIGHT_DEBUG_SCALE 1.0f
#endif

#ifndef LIGHT_CONF_FALLBACK
#define LIGHT_CONF_FALLBACK 0.65f
#endif

#ifndef LIGHT_CONF_CLAMPED_HIT
#define LIGHT_CONF_CLAMPED_HIT 0.75f
#endif

#ifndef LIGHT_NORMAL_VIEW_BLEND
#define LIGHT_NORMAL_VIEW_BLEND 0.20f
#endif

#ifndef LIGHT_NORMAL_GRAZING_START
#define LIGHT_NORMAL_GRAZING_START 0.35f
#endif

#ifndef LIGHT_NORMAL_GRAZING_END
#define LIGHT_NORMAL_GRAZING_END 0.05f
#endif

#ifndef LIGHT_CONF_LOW_ALPHA
#define LIGHT_CONF_LOW_ALPHA 0.65f
#endif

#ifndef LIGHT_CONF_GRAZING
#define LIGHT_CONF_GRAZING 0.75f
#endif

#ifndef LIGHT_CONF_DISP_K
#define LIGHT_CONF_DISP_K 8.0f
#endif

#ifndef LIGHT_HIT_UV_CLAMP
#define LIGHT_HIT_UV_CLAMP 1.5f
#endif

#ifndef LIGHT_HIT_DELTA2_MAX
#define LIGHT_HIT_DELTA2_MAX 0.05f
#endif

#ifndef LIGHT_ALPHA_SOFT_REF
#define LIGHT_ALPHA_SOFT_REF (2.0f / 255.0f)
#endif

#ifndef LIGHT_ALPHA_SKIP_THRESHOLD
#define LIGHT_ALPHA_SKIP_THRESHOLD (1.0f / 255.0f)
#endif

#ifndef LIGHT_NORMAL_CONF_BOOST
#define LIGHT_NORMAL_CONF_BOOST 0.35f
#endif

#ifndef LIGHT_NORMAL_ALPHA_BOOST
#define LIGHT_NORMAL_ALPHA_BOOST 0.30f
#endif

#ifndef LIGHT_WEAK_SHADE_REDUCTION
#define LIGHT_WEAK_SHADE_REDUCTION 0.75f
#endif

#ifndef LIGHT_DEEP_SHADE
#define LIGHT_DEEP_SHADE 0.2f   // deeper layers get 20% shading
#endif

#ifndef LIGHT_LDIR_BLEND
#define LIGHT_LDIR_BLEND 0.5f
#endif