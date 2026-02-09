#pragma once
#include "lighting_config.cuh"
#include "lighting_config_upload.cuh"

struct LightingOut {
    float diffuse_mul; // multiplied to base color
    float diffuse_base; // diffuse without energy compensation
    float spec_add;    // additive spec
    float lambert;     // lambert tern
    float ndotl;       // n*L
    float spot;        // spotlight factor
    float ambient;     // ambient value
    float kspec;       // specular factor
    float dkspecular;  // derivative of spec factor
    float spec_base;   // spec without ks
};

__device__ __forceinline__ float3 apply_norm_jacobian(float3 n_raw, float3 g_unit)
{
    // n_hat = n_raw / ||n_raw||
    float len2 = n_raw.x*n_raw.x + n_raw.y*n_raw.y + n_raw.z*n_raw.z;
    if (len2 <= 1e-20f) return make_float3(0.f, 0.f, 0.f);

    float inv_len = rsqrtf(len2);
    float3 n_hat = make_float3(n_raw.x * inv_len, n_raw.y * inv_len, n_raw.z * inv_len);

    // Project g_unit onto tangent plane
    float dotng = n_hat.x*g_unit.x + n_hat.y*g_unit.y + n_hat.z*g_unit.z;
    float3 g_proj = make_float3(
        g_unit.x - n_hat.x * dotng,
        g_unit.y - n_hat.y * dotng,
        g_unit.z - n_hat.z * dotng
    );
    // Normalize
    return make_float3(g_proj.x * inv_len, g_proj.y * inv_len, g_proj.z * inv_len);
}

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

__device__ __forceinline__ float ambient_value(const float* __restrict__ ambients,
                                                   const LightingConfig& cfg)
{
    if (cfg.ambient_mode == AMBIENT_OFF) {
        (void)ambients;
        return 0.0f;
    }
    if (cfg.ambient_mode == AMBIENT_FIXED) {
        (void)ambients;
        return cfg.ambient_fixed;
    }
    // learned
    return sigmoidf_stable(ambients[0]);
}

__device__ __forceinline__ float kspec_value(const float* __restrict__ kspecs,
                                                 const LightingConfig& cfg)
{
    if (cfg.phong_ks_mode == KS_LEARN) {
        return sigmoidf_stable(kspecs[0]); // [0,1]
    }
    return cfg.phong_ks;
}

__device__ __forceinline__ float shininess_value(float shininess,
                                                     const LightingConfig& cfg)
{
    if (cfg.phong_shiny_mode == SHINY_LEARN) {
        // typical: keep it positive and bounded (tweak range if needed)
        // maps R -> (0, +inf) but not too extreme
        return 1.0f + 63.0f * sigmoidf_stable(shininess); // [1,64] shininess[0]
    }
    return cfg.phong_shininess;
}

// Spotlight axis: camera forward (+Z)
__device__ __forceinline__ float spotlight_factor(const float3& light_dir_cam_to_surf,
                                                      const LightingConfig& cfg)
{
    if (cfg.use_spot == LIGHT_SPOT_OFF) {
        (void)light_dir_cam_to_surf;
        return 1.0f;
    }

    // axis: camera forward (+Z)
    const float3 axis = make_float3(0.f, 0.f, 1.f);
    float cosTheta = light_dir_cam_to_surf.x*axis.x
                   + light_dir_cam_to_surf.y*axis.y
                   + light_dir_cam_to_surf.z*axis.z;

    const float innerCos = cosf(cfg.spot_inner_deg * (LIGHT_PI / 180.f));
    const float outerCos = cosf(cfg.spot_outer_deg * (LIGHT_PI / 180.f));

    // avoid div0 if user sets equal angles
    float denom = fmaxf(innerCos - outerCos, 1e-6f);
    float t = (cosTheta - outerCos) / denom;

    float s = smoothstep01(t);
    if (cfg.spot_exp != 1.0f) s = powf(s, cfg.spot_exp);
    return s;
}

__device__ __forceinline__
LightingOut eval_lighting(
    const float2& pixf,
    int W, int H,
    float focal_x, float focal_y,
    float3 n_raw,
    const float* __restrict__ ambients,
    const float* __restrict__ kspecs,
    float shininess,
    const LightingConfig& cfg
) {
    LightingOut o;
    o.diffuse_mul  = 1.0f;
    o.diffuse_base = 1.0f;
    o.spec_add     = 0.0f;
    o.lambert      = 1.0f;
    o.ndotl        = 1.0f;
    o.spot         = 1.0f;
    o.ambient      = 0.0f;
    o.kspec        = 0.0f;
    o.dkspecular   = 0.0f;
    o.spec_base    = 0.0f;

    const bool useLambert =
        (cfg.light_mode == LAMBERT_ONLY) ||
        (cfg.light_mode == LAMBERT_PHONG);

    const bool usePhong =
        (cfg.light_mode == PHONG_ONLY) ||
        (cfg.light_mode == LAMBERT_PHONG);

    // Default, if neither is used
    if (!(useLambert || usePhong)) {
        return o;
    }

    // Normalize normal
    float3 n = normalize_or_default(n_raw, make_float3(0.f,0.f,1.f));

    // camera->surface
    float3 light_dir = compute_light_dir(pixf, W, H, focal_x, focal_y);
    light_dir = normalize_or_default(light_dir, make_float3(0.f,0.f,1.f));

    // surface->camera
    float3 V = make_float3(-light_dir.x, -light_dir.y, -light_dir.z);
    V = normalize_or_default(V, make_float3(0.f,0.f,-1.f));

    // flashlight at camera
    float3 L = V;
    L = normalize_or_default(L, make_float3(0.f,0.f,-1.f));

    const float spot = spotlight_factor(light_dir, cfg);
    o.spot = spot;

    const float ndotl = n.x*L.x + n.y*L.y + n.z*L.z;
    o.ndotl = ndotl;

    float lambert = 1.0f;
    if (useLambert) {
        if (cfg.lambert_mode == LAMBERT_ABS) {
            lambert = fabsf(ndotl);
        } else {
            lambert = fmaxf(ndotl, 0.0f);
        }
    }
    o.lambert = lambert;

    // Ambient
    float a = ambient_value(ambients, cfg);
    o.ambient = a;

    // Diffuse
    float diffuse = 1.0f; // multiplied to base color
    if (useLambert) {
        diffuse = a + (1.0f - a) * lambert * spot;
    } else {
        // Phong-only: base color
        diffuse = 1.0f;
    }
    o.diffuse_base = diffuse;

    // Specular
    float spec = 0.0f;

    if (usePhong) {
        float3 Hh = make_float3(L.x + V.x, L.y + V.y, L.z + V.z);
        Hh = normalize_or_default(Hh, make_float3(0.f,0.f,-1.f));

        float ndoth = fmaxf(n.x*Hh.x + n.y*Hh.y + n.z*Hh.z, 0.0f);

        float shiny = shininess_value(shininess, cfg);
        float specular = (ndoth > 0.0f) ? powf(ndoth, shiny) : 0.0f;

        float ks = kspec_value(kspecs, cfg);

        // derivative for sigmoid parameter if KS learned
        float dkspecular = 0.0f;
        if (cfg.phong_ks_mode == KS_LEARN) {
            dkspecular = ks * (1.0f - ks);
        }

        float spec_base = specular * spot;

        // gating
        if (cfg.spec_gating == SPEC_GATING_BACKFACE) {
            if (ndotl <= 0.0f) spec_base = 0.0f;
        } else if (cfg.spec_gating == SPEC_GATING_LAMBERTSCALE) {
            // only meaningful if lambert is computed; if not, lambert==1 above
            spec_base *= lambert;
        }

        spec = ks * spec_base;

        // energy compensation
        if (cfg.energy_comp == ENERGY_COMP_ON && useLambert) {
            diffuse *= (1.0f - ks);
        }

        o.kspec = ks;
        o.dkspecular = dkspecular;
        o.spec_base = spec_base;
    } else {
        o.kspec      = 0.0f;
        o.dkspecular = 0.0f;
        o.spec_base  = 0.0f;
    }

    // Final output
    o.diffuse_mul = diffuse;
    o.spec_add = spec;
    return o;
}