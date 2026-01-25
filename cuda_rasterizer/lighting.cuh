#pragma once
#include "lighting_config.cuh"

struct LightingOut {
    float diffuse_mul; // multiplied to base color
    float spec_add;    // additive spec
    float lambert;     // lambert tern
    float ndotl;       // n*L
    float spot;        // spotlight factor
    float ambient;     // ambient value
};

__device__ __forceinline__
float3 compute_light_dir(const float2& pixf,
                                   int W, int H,
                                   float focal_x, float focal_y)
{
    float x = (pixf.x - 0.5f * (float)W) / focal_x;
    float y = (pixf.y - 0.5f * (float)H) / focal_y;
    float3 v = make_float3(x, y, 1.0f);

    float len2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (len2 < 1e-20f) return make_float3(0.f, 0.f, 1.f);

    float inv_len = rsqrtf(len2);
    v.x *= inv_len;
    v.y *= inv_len;
    v.z *= inv_len;
    return v;
}

__device__ __forceinline__ float sigmoidf_stable(float x)
{
    // stable sigmoid avoids exp overflow
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

__device__ __forceinline__ float3 normalize_or_default(float3 v, float3 def) {
    float len2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (len2 <= 1e-20f) return def;
    float inv = rsqrtf(len2);
    v.x *= inv; v.y *= inv; v.z *= inv;
    return v;
}

__device__ __forceinline__ float clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

__device__ __forceinline__ float smoothstep01(float t) {
    t = clamp01(t);
    return t*t*(3.0f - 2.0f*t);
}

__device__ __forceinline__ float ambient_value(const float* __restrict__ ambients) {
#if (LIGHT_AMBIENT_MODE == 0)
    (void)ambients;
    return 0.0f;
#elif (LIGHT_AMBIENT_MODE == 1)
    (void)ambients;
    return LIGHT_AMBIENT_FIXED;
#else
    return sigmoidf_stable(ambients[0]);
#endif
}

// Spotlight axis: camera forward (+Z).
__device__ __forceinline__ float spotlight_factor(const float3& light_dir_cam_to_surf) {
#if LIGHT_USE_SPOT
    const float3 axis = make_float3(0.f, 0.f, 1.f);
    float cosTheta = light_dir_cam_to_surf.x*axis.x + light_dir_cam_to_surf.y*axis.y + light_dir_cam_to_surf.z*axis.z;

    const float innerCos = cosf(LIGHT_SPOT_INNER_DEG * (LIGHT_PI / 180.f));
    const float outerCos = cosf(LIGHT_SPOT_OUTER_DEG * (LIGHT_PI / 180.f));

    float t = (cosTheta - outerCos) / (innerCos - outerCos);
    float s = smoothstep01(t);
    if (LIGHT_SPOT_EXP != 1.0f) s = powf(s, LIGHT_SPOT_EXP);
    return s;
#else
    (void)light_dir_cam_to_surf;
    return 1.0f;
#endif
}

__device__ __forceinline__
LightingOut eval_lighting(
    const float2& pixf,
    int W, int H,
    float focal_x, float focal_y,
    float3 n_raw,
    const float* __restrict__ ambients
) {
    LightingOut o;
    o.diffuse_mul = 1.0f;
    o.spec_add    = 0.0f;
    o.lambert     = 1.0f;
    o.ndotl       = 1.0f;
    o.spot        = 1.0f;
    o.ambient     = 0.0f;

#if (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
    float3 n = normalize_or_default(n_raw, make_float3(0.f,0.f,1.f));

    float3 light_dir = compute_light_dir(pixf, W, H, focal_x, focal_y);         // camera->surface
    light_dir = normalize_or_default(light_dir, make_float3(0.f,0.f,1.f));

    float3 V = make_float3(-light_dir.x, -light_dir.y, -light_dir.z);           // surface->camera
    V = normalize_or_default(V, make_float3(0.f,0.f,-1.f));

    float3 L = V;                                                               // flashlight at camera
    L = normalize_or_default(L, make_float3(0.f,0.f,-1.f));

    float spot = spotlight_factor(light_dir);
    o.spot = spot;

    float ndotl = n.x*L.x + n.y*L.y + n.z*L.z;
    o.ndotl = ndotl;

#if LIGHT_USE_LAMBERT
  #if LIGHT_LAMBERT_ABS
    float lambert = fabsf(ndotl);
  #else
    float lambert = fmaxf(ndotl, 0.0f);
  #endif
#else
    float lambert = 1.0f;
#endif
    o.lambert = lambert;

    float a = ambient_value(ambients);
    o.ambient = a;

#if LIGHT_USE_LAMBERT
    float diffuse = a + (1.0f - a) * lambert * spot;
#else
    // Phong-only: keeping a neutral base
    float diffuse = 1.0f;
#endif

#if LIGHT_USE_PHONG
    float3 Hh = make_float3(L.x + V.x, L.y + V.y, L.z + V.z);
    Hh = normalize_or_default(Hh, make_float3(0.f,0.f,-1.f));
    float ndoth = fmaxf(n.x*Hh.x + n.y*Hh.y + n.z*Hh.z, 0.0f);
    float specular = (ndoth > 0.0f) ? powf(ndoth, LIGHT_PHONG_SHININESS) : 0.0f;

    float spec = LIGHT_PHONG_KS * specular * spot;

  #if (LIGHT_SPEC_GATING == 1)
    if (ndotl <= 0.0f) spec = 0.0f;
  #elif (LIGHT_SPEC_GATING == 2)
    spec *= lambert;
  #endif

#else
    float spec = 0.0f;
#endif

#if LIGHT_ENERGY_COMP
  #if LIGHT_USE_LAMBERT
    diffuse *= (1.0f - LIGHT_PHONG_KS);
  #endif
#endif

    o.diffuse_mul = diffuse;
    o.spec_add    = spec;
#endif

    return o;
}