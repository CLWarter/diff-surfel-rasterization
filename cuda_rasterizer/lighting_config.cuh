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

#ifndef LIGHT_PHONG_KS
#define LIGHT_PHONG_KS 0.10f
#endif

#ifndef LIGHT_PHONG_SHININESS
#define LIGHT_PHONG_SHININESS 8.0f
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
#define LIGHT_SPOT_INNER_DEG 15.0f
#endif

#ifndef LIGHT_SPOT_OUTER_DEG
#define LIGHT_SPOT_OUTER_DEG 35.0f
#endif

// smooth ramp
#ifndef LIGHT_SPOT_EXP
#define LIGHT_SPOT_EXP 1.5f
#endif

// Light constants
#ifndef LIGHT_PI
#define LIGHT_PI 3.14159265358979323846f
#endif