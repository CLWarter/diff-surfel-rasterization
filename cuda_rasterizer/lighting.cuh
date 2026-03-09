#pragma once
#include "lighting_config.cuh"

struct LightingOut {
    float diffuse_mul;  // multiplied to base color
    float diffuse_base; // diffuse without energy compensation

    float spec_add;    // additive spec
    float spec_base;   // spec without ks
    float spec_pow;    // ndot^shiny

    float lambert;     // lambert term
    float ndotl;       // n*L
    float ndoth;       // n*H

    float spot;        // spotlight factor

    float ambient;     // ambient value

    float intensity;            // Li = I * inv
    float inv;                  // store inv for backward
    float dintensity_ddepth;    // d(Li)/d(depth_cam)
    float li_clamped;           // 1 if Li was clamped, else 0

    float dI_raw;      // dI / dI_raw (sigmoid mapping)

    float kspec;       // specular factor
    float dkspecular;  // derivative of spec factor

    float shiny;       // shininess exponent
    float dshin_raw;   // d(shiny)/d(shiny_raw)

};

__device__ __forceinline__ float3 apply_norm_jacobian(float3 n_raw, float3 g_unit)
{
    const float eps_len2 = 1e-6f; // prevent blow-up
    float len2 = n_raw.x*n_raw.x + n_raw.y*n_raw.y + n_raw.z*n_raw.z;

    if (len2 <= eps_len2) return make_float3(0.f, 0.f, 0.f);

    float inv_len = rsqrtf(len2);
    float3 n_hat = make_float3(n_raw.x * inv_len, n_raw.y * inv_len, n_raw.z * inv_len);

    // Project g_unit onto tangent plane
    float dotng = n_hat.x*g_unit.x + n_hat.y*g_unit.y + n_hat.z*g_unit.z;
    float3 g_proj = make_float3(
        g_unit.x - n_hat.x * dotng,
        g_unit.y - n_hat.y * dotng,
        g_unit.z - n_hat.z * dotng
    );
    // extra clamp: prevents rare spikes even when len2 is barely above eps
    const float inv_len_max = 100.0f;
    float inv_len_clamped = fminf(inv_len, inv_len_max);
    // Normalize
    return make_float3(g_proj.x * inv_len_clamped, g_proj.y * inv_len_clamped, g_proj.z * inv_len_clamped);
}

__device__ __forceinline__
float3 compute_light_dir(const float2& pixf,
                                   int W, int H,
                                   float focal_x, float focal_y, float depth_cam)
{
    // --- camera ray (unnormalized) ---
    float x = (pixf.x - 0.5f * (float)W) / focal_x;
    float y = (pixf.y - 0.5f * (float)H) / focal_y;

    float3 ray = make_float3(x, y, 1.0f);

    // --- surface point in camera space ---
    float3 P = make_float3(depth_cam * ray.x,
                           depth_cam * ray.y,
                           depth_cam);

    // --- light position: 1cm left + up (diagonal) ---
    const float comp = 0.01f * 0.70710678f;  // 1cm / sqrt(2)

    const float3 light_pos = make_float3(
        -comp,   // left (negative x)
        -comp,   // up   (negative y in your pixel convention)
         0.0f
    );

    // --- vector surface -> light ---
    float3 S21 = make_float3(light_pos.x - P.x,
                             light_pos.y - P.y,
                             light_pos.z - P.z);

    float len2 = S21.x*S21.x + S21.y*S21.y + S21.z*S21.z;
    if (len2 < 1e-20f)
        return make_float3(0.f, 0.f, -1.f);

    float inv_len = rsqrtf(len2);
    S21.x *= inv_len;
    S21.y *= inv_len;
    S21.z *= inv_len;

    return S21;  // normalized surface -> light direction
}

__device__ __forceinline__ float distance_attenuation(float d, int mode, float k) {
    if (mode == 0) return 1.0f;
    // mode 1: quadratic
    return 1.0f / (1.0f + k * d * d);
}

__device__ __forceinline__ float inv_quadratic_falloff(float d)
{
    // Hardcode for now (your “no dynamic config” branch)
    // k controls how fast it falls off; tune later
    const float k = 0.15f;
    return 1.0f / (1.0f + k * d * d);
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
    const float amax = 0.25f;
    float t = sigmoidf_stable(ambients[0]);
    return amax * t;
#endif
}

__device__ __forceinline__
float kspec_value(const float* kspecs)
{
#if (LIGHT_PHONG_KS_MODE == 1)
    // learned scalar value between [0,1]
    return sigmoidf_stable(kspecs[0]);
#else
    return LIGHT_PHONG_KS;
#endif
}

__device__ __forceinline__ float shininess_value(const float* shiny_raw, float* dshin_draw_out)
{
#if (LIGHT_PHONG_SHININESS_MODE == 1)
    float t = sigmoidf_stable(shiny_raw[0]);  // in (0,1)
    if (dshin_draw_out)
        *dshin_draw_out = (LIGHT_SHINY_MAX - LIGHT_SHINY_MIN) * t * (1.0f - t);
    return LIGHT_SHINY_MIN + (LIGHT_SHINY_MAX - LIGHT_SHINY_MIN) * t;
#else
    if (dshin_draw_out) *dshin_draw_out = 0.0f;
    return LIGHT_PHONG_SHININESS;
#endif
}

// Spotlight axis: camera forward (+Z)
__device__ __forceinline__ float spotlight_factor(const float3& light_dir_cam_to_surf) {
#if LIGHT_USE_SPOT
    const float3 axis = make_float3(0.f, 0.f, 1.f);
    float cosTheta = light_dir_cam_to_surf.x*axis.x + light_dir_cam_to_surf.y*axis.y + light_dir_cam_to_surf.z*axis.z;

    const float innerCos = cosf(LIGHT_SPOT_INNER_DEG * (LIGHT_PI / 180.f));
    const float outerCos = cosf(LIGHT_SPOT_OUTER_DEG * (LIGHT_PI / 180.f));

    float denom = fmaxf(innerCos - outerCos, 1e-6f);
    float t = (cosTheta - outerCos) / denom;
    float s = smoothstep01(t);
    if (LIGHT_SPOT_EXP != 1.0f) s = powf(s, LIGHT_SPOT_EXP);
    return s;
#else
    (void)light_dir_cam_to_surf;
    return 1.0f;
#endif
}

__device__ __forceinline__ float intensity_value(
    const float* __restrict__ intensity_raw,
    float* dI_draw_out // returns dI/d(intensity_raw[0]) when learnable, else 0
) {
#if (LIGHT_INTENSITY_MODE == 1)
    // learnable: map sigmoid(raw) into [MIN, MAX]
    float t = sigmoidf_stable(intensity_raw[0]); // in (0,1)
    if (dI_draw_out)
        *dI_draw_out = (LIGHT_INTENSITY_MAX - LIGHT_INTENSITY_MIN) * t * (1.0f - t);
    return LIGHT_INTENSITY_MIN + (LIGHT_INTENSITY_MAX - LIGHT_INTENSITY_MIN) * t;
#else
    // constant
    if (dI_draw_out) *dI_draw_out = 0.0f;
    (void)intensity_raw;
    return LIGHT_INTENSITY_CONST;
#endif
}

__device__ __forceinline__
float3 compute_ray_unnorm(const float2& pixf, int W, int H, float focal_x, float focal_y)
{
    float x = (pixf.x - 0.5f * (float)W) / focal_x;
    float y = (pixf.y - 0.5f * (float)H) / focal_y;
    return make_float3(x, y, 1.0f); // unnormalized
}

__device__ __forceinline__
LightingOut eval_lighting(
    const float2& pixf,
    int W, int H,
    float focal_x, float focal_y,
    float3 n_raw,
    float depth_cam,
    const float* __restrict__ ambients,
    const float* __restrict__ intensity,
    const float* __restrict__ kspecs,
    const float* __restrict__ shiny
) {
    LightingOut o = {};
    // defaults (no lighting)
    o.diffuse_mul = 1.0f;
    o.diffuse_base = 1.0f;
    o.spec_add    = 0.0f;
    o.spec_base   = 0.0f;
    o.spec_pow    = 0.0f;
    o.lambert     = 1.0f;
    o.ndotl       = 1.0f;
    o.ndoth       = 0.0f;
    o.spot        = 1.0f;
    o.ambient     = 0.0f;
    o.intensity   = 1.0f;
    o.inv         = 1.0f;
    o.dintensity_ddepth = 0.0f;
    o.dI_raw      = 0.0f;
    o.li_clamped = 0.0f;
    o.kspec       = 0.0f;
    o.dkspecular  = 0.0f;
    o.shiny       = LIGHT_PHONG_SHININESS;
    o.dshin_raw   = 0.0f;


    // light direction
#if (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
    float3 n = normalize_or_default(n_raw, make_float3(0.f,0.f,1.f));

    // camera->surface direction (for view + spotlight cone)
    float3 view_ray = normalize_or_default(
        make_float3((pixf.x - 0.5f * W) / focal_x,
                    (pixf.y - 0.5f * H) / focal_y,
                    1.0f),
        make_float3(0.f,0.f,1.f));

    // surface->light direction
    float3 L = compute_light_dir(pixf, W, H, focal_x, focal_y, depth_cam);
    L = normalize_or_default(L, make_float3(0.f,0.f,-1.f));

    // surface->camera (view) direction
    float3 V = make_float3(-view_ray.x, -view_ray.y, -view_ray.z);
    V = normalize_or_default(V, make_float3(0.f,0.f,-1.f));

    // spotlight cone should depend on camera->surface direction, not surface->light
    float spot = spotlight_factor(view_ray);
    o.spot = spot;

    float ndotl = n.x*L.x + n.y*L.y + n.z*L.z;
    o.ndotl = ndotl;

    // lambert
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

    // ambient
    float a = ambient_value(ambients);
    o.ambient = a;

    // intensity
    float dI_draw = 0.0f;
    float I = intensity_value(intensity, &dI_draw);
    o.dI_raw = dI_draw; // dI/d(raw) if learnable, else 0

    // ---------------- distance / falloff with offset light ----------------

    // Ray through pixel in camera coords (z=1, unnormalized)
    float3 r = compute_ray_unnorm(pixf, W, H, focal_x, focal_y);

    // Camera-space hit point at z = depth_cam
    float3 P = make_float3(depth_cam * r.x,
                        depth_cam * r.y,
                        depth_cam);

    // Light position in camera space (1cm left + up, diagonal)
    const float comp = 0.01f * 0.70710678f;
    const float3 light_pos = make_float3(-comp, -comp, 0.0f);

    // Vector from light to point (L -> P)
    float3 LP = make_float3(P.x - light_pos.x,
                            P.y - light_pos.y,
                            P.z - light_pos.z);

    // -------- distance falloff (for Li) --------
    float inv = 1.0f;
    o.dintensity_ddepth = 0.0f;

    // dist^2 and dist
    float dist2 = fmaxf(LP.x*LP.x + LP.y*LP.y + LP.z*LP.z, 1e-12f);

    #if (FALLOFF_MODE == 0)
        // no falloff
        inv = 1.0f;
        o.dintensity_ddepth = 0.0f;

    #elif (FALLOFF_MODE == 1)
        // quadratic-ish: inv = 1/(1 + k*dist^2)
        const float k = FALLOFF_K;
        inv = 1.0f / (1.0f + k * dist2);

        // Derivative for backward: d(Li)/d(depth)
        float d_dist2_ddepth = 2.0f * (LP.x*r.x + LP.y*r.y + LP.z*1.0f);
        float d_inv_d_dist2  = -k * inv * inv;
        o.dintensity_ddepth  = I * d_inv_d_dist2 * d_dist2_ddepth;

    #else
        // fallback: treat as no falloff
        inv = 1.0f;
        o.dintensity_ddepth = 0.0f;
    #endif

    o.inv = inv;

    // Li_raw = I * inv
    float Li_raw = I * inv;
    float Li = Li_raw;

    #if (LIGHT_LI_CLAMP > 0)
        if (Li > (float)LIGHT_LI_CLAMP) {
            Li = (float)LIGHT_LI_CLAMP;
            o.li_clamped = 1.0f;
        }
    #endif

    o.intensity = Li;

    // kspec / shiny
#if LIGHT_USE_PHONG
    float ks = kspec_value(kspecs);

    float dkspecular = 0.0f;
  #if (LIGHT_PHONG_KS_MODE == 1)
    dkspecular = ks * (1.0f - ks);
    #else
    dkspecular = 0.0f;
  #endif

    float dshin_draw = 0.0f;
    float shin = shininess_value(shiny, &dshin_draw);

    o.kspec = ks;
    o.dkspecular = dkspecular;
    o.shiny = shin;
    o.dshin_raw = dshin_draw;
#endif

    // diffuse multiplier
#if LIGHT_USE_LAMBERT
        float diffuse_base = a + (1.0f - a) * lambert * spot * Li;
        o.diffuse_base = diffuse_base;

    #if LIGHT_ENERGY_COMP && LIGHT_USE_PHONG
        diffuse_base *= (1.0f - o.kspec);
    #endif
        o.diffuse_mul = diffuse_base;
#endif

    // phong spec
#if LIGHT_USE_PHONG
    float3 Hh = normalize_or_default(make_float3(L.x + V.x, L.y + V.y, L.z + V.z),
                                     make_float3(0.f,0.f,-1.f));
    float ndoth = fmaxf(n.x*Hh.x + n.y*Hh.y + n.z*Hh.z, 0.0f);
    o.ndoth = ndoth;

    float spec_pow = (ndoth > 0.0f) ? powf(ndoth, o.shiny) : 0.0f;
    o.spec_pow = spec_pow;

    float spec_base = spec_pow * spot * Li;

  #if (LIGHT_SPEC_GATING == 1)
    if (ndotl <= 0.0f) spec_base = 0.0f;
  #elif (LIGHT_SPEC_GATING == 2)
    spec_base *= o.lambert;
  #endif

    o.spec_base = spec_base;
    float spec_add = o.kspec * spec_base;

    o.spec_add = spec_add;

#endif

#endif
    return o;
}