#pragma once
#include "lighting_config.cuh"

struct LightingOut {
    // material-only BRDF terms
    // diffuse_rgb = kD / pi   (no light, no ambient)
    // spec_rgb    = GGX Cook-Torrance BRDF term (no light)
    // brdf_rgb    = diffuse_rgb + spec_rgb
    float3 diffuse_rgb;
    float3 spec_rgb;
    float3 brdf_rgb;

    // shading decomposition used by the renderer
    // indirect_approx_rgb is NOT part of the BRDF.
    // It is only an approximate indirect ambient irradiance multiplier.
    float3 indirect_approx_rgb;   // scalar ambient expanded to RGB multiplier
    float3 direct_diffuse_rgb;    // diffuse BRDF after direct light evaluation

    // angular/light terms
    float lambert;
    float ndotl, ndotv, ndoth, vdoth;
    float spot;
    float inv;
    float Li;        // final intensity
    float I;
    float li_clamped;
    float Li_raw;

    float metallic;
    float roughness;
    float alpha;      // roughness^2
    float alpha2;
    float F0;

    float D;
    float G;
    float Gv;
    float Gl;
    float fresnel;

    float dI_raw;
    float dmetal_raw;
    float drough_raw;

    float diffuse_mul;   // legacy scalar proxy
    float3 diffuse_mul_rgb; // per-channel diffuse multiplier
    float spec_add;      // legacy scalar alias = luminance-ish proxy of RGB spec
    float3 spec_add_rgb; // final additive RGB specular

    // diffuse decomposition
    float diffuse_brdf;     // e.g. 1/pi for Lambert
    float indirect_diffuse; // ambient / indirect approximation
    float direct_diffuse_raw; // brdf * lambert * spot * Li
    float direct_diffuse;     // after optional energy compensation

    // legacy-compatible aliases
    float diffuse_amb;      // same as indirect_diffuse
    float diffuse_dir_raw;  // same as direct_diffuse_raw
    float diffuse_dir;      // same as direct_diffuse

    // spec decomposition
    float spec_pow;          // legacy alias: stores D term
    float spec_dir_raw;      // legacy scalar proxy
    float spec_dir_gated;    // legacy scalar proxy
    float spec_base;         // legacy scalar proxy

    float3 spec_dir_raw_rgb;
    float3 spec_dir_gated_rgb;
    float3 spec_base_rgb;

    // ambient/intensity
    float ambient;
    float intensity;         // Li = I * inv 
    float dintensity_ddepth;

    float3 F0_rgb;
    float3 fresnel_rgb;
};

/* DEPRECATED */
/* __device__ __forceinline__
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

    const float3 light_pos = make_float3(
         0.0f,   // left (negative x)
         0.0f,   // up   (negative y in your pixel convention)
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
} */

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

__device__ __forceinline__ float softplusf_stable(float x)
{
    // numerically stable softplus
    if (x > 20.0f) return x;          // log(1 + exp(x)) ~ x
    if (x < -20.0f) return expf(x);   // log(1 + exp(x)) ~ exp(x)
    return log1pf(expf(x));
}

__device__ __forceinline__ float softplus_beta2(float x)
{
    const float beta = 2.0f;

    float bx = beta * x;

    // numerically stable
    if (bx > 20.0f)
        return x;                   // ≈ x
    if (bx < -20.0f)
        return expf(bx) / beta;     // ≈ exp(2x)/2

    return log1pf(expf(bx)) / beta;
}

__device__ __forceinline__ float3 normalize_or_default(float3 v, float3 def) {
    float len2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (len2 <= 1e-20f) return def;
    float inv = rsqrtf(len2);
    v.x *= inv; v.y *= inv; v.z *= inv;
    return v;
}

__device__ __forceinline__ float clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

__device__ __forceinline__
float saturate01(float x)
{
    return fmaxf(0.0f, fminf(x, 1.0f));
}

__device__ __forceinline__ float smoothstep01(float t) {
    t = saturate01(t);
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
    float t = sigmoidf_stable(ambients[0]);
    return LIGHT_AMBIENT_MAX * t;
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

__device__ __forceinline__ float shininess_value(const float* metallic_raw, float* dshin_draw_out)
{
#if (LIGHT_PHONG_SHININESS_MODE == 1)
    float t = sigmoidf_stable(metallic_raw[0]);  // in (0,1)
    if (dshin_draw_out)
        *dshin_draw_out = (LIGHT_SHINY_MAX - LIGHT_SHINY_MIN) * t * (1.0f - t);
    return LIGHT_SHINY_MIN + (LIGHT_SHINY_MAX - LIGHT_SHINY_MIN) * t;
#else
    if (dshin_draw_out) *dshin_draw_out = 0.0f;
    return LIGHT_PHONG_SHININESS;
#endif
}

__device__ __forceinline__ float roughness_value(
    const float* roughness_raw,
    float* drough_draw_out)
{
#if (LIGHT_GGX_ROUGHNESS_MODE == 1)
    float t = sigmoidf_stable(roughness_raw[0]);

    const float rmin = LIGHT_GGX_ROUGHNESS_MIN;
    const float rmax = LIGHT_GGX_ROUGHNESS_MAX;
    const float range = rmax - rmin;

    if (drough_draw_out)
        *drough_draw_out = range * t * (1.0f - t);

    return rmin + range * t;
#else
    (void)roughness_raw;

    if (drough_draw_out)
        *drough_draw_out = 0.0f;

    return fminf(
        fmaxf(LIGHT_GGX_ROUGHNESS, LIGHT_GGX_ROUGHNESS_MIN),
        LIGHT_GGX_ROUGHNESS_MAX
    );
#endif
}

__device__ __forceinline__ float metallic_value(const float* metallic_raw, float* dmetal_draw_out)
{
#if (LIGHT_GGX_METALLIC_MODE == 1)
    float t = sigmoidf_stable(metallic_raw[0]);
    if (dmetal_draw_out)
        *dmetal_draw_out = (LIGHT_GGX_METALLIC_MAX - LIGHT_GGX_METALLIC_MIN) * t * (1.0f - t);
    return LIGHT_GGX_METALLIC_MIN + (LIGHT_GGX_METALLIC_MAX - LIGHT_GGX_METALLIC_MIN) * t;
#else
    (void)metallic_raw;
    if (dmetal_draw_out) *dmetal_draw_out = 0.0f;
    return fminf(fmaxf(LIGHT_GGX_METALLIC, LIGHT_GGX_METALLIC_MIN), LIGHT_GGX_METALLIC_MAX);
#endif
}

__device__ __forceinline__ float fresnel_schlick_scalar(float VdotH, float F0)
{
    float x = 1.0f - fmaxf(VdotH, 0.0f);
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return F0 + (1.0f - F0) * x5;
}

__device__ __forceinline__ float3 fresnel_schlick_rgb(float VdotH, const float3& F0)
{
    float x = 1.0f - fmaxf(VdotH, 0.0f);
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return make_float3(
        F0.x + (1.0f - F0.x) * x5,
        F0.y + (1.0f - F0.y) * x5,
        F0.z + (1.0f - F0.z) * x5
    );
}

__device__ __forceinline__ float ggx_D(float NdotH, float alpha2)
{
    float nh = fmaxf(NdotH, 0.0f);
    float denom = nh * nh * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (LIGHT_PI * denom * denom + LIGHT_GGX_DENOM_EPS);
}

__device__ __forceinline__ float ggx_D_and_dDdnh(
    float NdotH, float alpha2,
    float* dD_dnh_out)   // pass nullptr if derivative not needed
{
    float nh    = fmaxf(NdotH, 0.0f);
    float t     = nh * nh * (alpha2 - 1.0f) + 1.0f;
    float Dden  = LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS;
    float D     = alpha2 / Dden;

    if (dD_dnh_out)
        *dD_dnh_out = (-4.0f * LIGHT_PI * alpha2 * nh * (alpha2 - 1.0f) * t)
                      / fmaxf(Dden * Dden, 1e-12f);
    return D;
}

__device__ __forceinline__ float smith_G1_schlick_ggx(float NdotX, float roughness)
{
    float nx = fmaxf(NdotX, 0.0f);
    float r = roughness;
    float k = ((r + 1.0f) * (r + 1.0f)) * 0.125f; // (r+1)^2 / 8
    return nx / (nx * (1.0f - k) + k + LIGHT_GGX_DENOM_EPS);
}

__device__ __forceinline__ float ggx_specular_bridge(
    float NdotL,
    float NdotV,
    float NdotH,
    float VdotH,
    float roughness,
    float F0,
    float* D_out = nullptr,
    float* G_out = nullptr,
    float* F_out = nullptr,
    float* Gv_out = nullptr,
    float* Gl_out = nullptr)
{
    float nv = fmaxf(NdotV, LIGHT_GGX_NV_EPS);
    float nl = fmaxf(NdotL, LIGHT_GGX_NL_EPS);

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float D = ggx_D(NdotH, alpha2);
    float Gv = smith_G1_schlick_ggx(nv, roughness);
    float Gl = smith_G1_schlick_ggx(nl, roughness);
    float G = Gv * Gl;
    float F = fresnel_schlick_scalar(VdotH, F0);

    if (D_out)  *D_out = D;
    if (G_out)  *G_out = G;
    if (F_out)  *F_out = F;
    if (Gv_out) *Gv_out = Gv;
    if (Gl_out) *Gl_out = Gl;

    return (D * G * F) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);
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
        float raw0 = intensity_raw[0];
        float raw = fminf(fmaxf(raw0, -15.0f), 15.0f);

        float I = softplus_beta2(raw);

        float gate = (raw0 >= -15.0f && raw0 <= 15.0f) ? 1.0f : 0.0f;
        if (dI_draw_out)
            *dI_draw_out = gate * sigmoidf_stable(2.0f * raw);

        return I;
    #else
        if (dI_draw_out) *dI_draw_out = 0.0f;
        (void)intensity_raw;
        return LIGHT_INTENSITY_CONST;
    #endif
}

__device__ __forceinline__ float3 float3_add(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 float3_mul(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 float3_scale(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float float3_avg(const float3& a)
{
    return (a.x + a.y + a.z) * (1.0f / 3.0f);
}


struct LightMaterialValues
{
    float metallic;
    float roughness;
    float dmetal_draw;
    float drough_draw;
};

__device__ __forceinline__
LightMaterialValues eval_light_material_values(
    const float* __restrict__ metallic_raw,
    const float* __restrict__ roughness_raw)
{
    LightMaterialValues m;

    m.dmetal_draw = 0.0f;
    m.drough_draw = 0.0f;

#if (LIGHT_GGX_METALLIC_MODE == 1)
    if (metallic_raw != nullptr)
        m.metallic = metallic_value(metallic_raw, &m.dmetal_draw);
    else
    {
        m.metallic = LIGHT_GGX_METALLIC;
        m.dmetal_draw = 0.0f;
    }
#else
    m.metallic = metallic_value(nullptr, &m.dmetal_draw);
#endif

#if (LIGHT_GGX_ROUGHNESS_MODE == 1)
    if (roughness_raw != nullptr)
        m.roughness = roughness_value(roughness_raw, &m.drough_draw);
    else
    {
        m.roughness = LIGHT_GGX_ROUGHNESS;
        m.drough_draw = 0.0f;
    }
#else
    m.roughness = roughness_value(nullptr, &m.drough_draw);
#endif

    m.metallic = saturate01(m.metallic);
    m.roughness = saturate01(m.roughness);

    return m;
}

__device__ __forceinline__
float3 faceforward_basis_normal(
    const float3& bu_cam,
    const float3& bv_cam,
    const float3& point_cam)
{
    float3 n_basis = cross(bu_cam, bv_cam);
    n_basis = normalize_or_default(n_basis, make_float3(0.0f, 0.0f, 1.0f));

    float3 view_ray = normalize_or_default(point_cam, make_float3(0.0f, 0.0f, 1.0f));
    float3 V = make_float3(-view_ray.x, -view_ray.y, -view_ray.z);

    float ndotv =
        n_basis.x * V.x +
        n_basis.y * V.y +
        n_basis.z * V.z;

    if (ndotv < 0.0f)
    {
        n_basis.x = -n_basis.x;
        n_basis.y = -n_basis.y;
        n_basis.z = -n_basis.z;
    }

    return n_basis;
}

__device__ __forceinline__
float3 diffuse_kd_from_fresnel_metallic(
    const float3& fresnel_rgb,
    float metallic)
{
    return make_float3(
        (1.0f - fresnel_rgb.x) * (1.0f - metallic),
        (1.0f - fresnel_rgb.y) * (1.0f - metallic),
        (1.0f - fresnel_rgb.z) * (1.0f - metallic)
    );
}

__device__ __forceinline__
float3 compute_ray_unnorm(const float2& pixf, int W, int H, float focal_x, float focal_y)
{
    float x = (pixf.x - 0.5f * (float)W) / focal_x;
    float y = (pixf.y - 0.5f * (float)H) / focal_y;
    return make_float3(x, y, 1.0f); // unnormalized
}


__device__ __forceinline__
LightingOut eval_lighting_surface_values(
    const float2& pixf,
    int W, int H,
    float focal_x, float focal_y,
    float3 normal_raw,
    float depth_cam,
    const float* __restrict__ ambients,
    const float* __restrict__ intensity,
    float roughness_in,
    float metallic_in,
    const float3& base_color,
    const float3* point_cam_opt = nullptr)
{
    float rough_local = roughness_in;
    float metal_local = metallic_in;

    LightingOut o = {};

    o.diffuse_mul_rgb = make_float3(0.0f, 0.0f, 0.0f);
    o.spec_add_rgb = make_float3(0.0f, 0.0f, 0.0f);

#if (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
    float3 n = normalize_or_default(normal_raw, make_float3(0.0f, 0.0f, 1.0f));

    float3 P;
    float3 view_ray;

    if (point_cam_opt != nullptr)
    {
        P = *point_cam_opt;
        view_ray = normalize_or_default(P, make_float3(0.0f, 0.0f, 1.0f));
    }
    else
    {
        view_ray = normalize_or_default(
            make_float3((pixf.x - 0.5f * W) / focal_x,
                        (pixf.y - 0.5f * H) / focal_y,
                        1.0f),
            make_float3(0.0f, 0.0f, 1.0f));

        float3 r = compute_ray_unnorm(pixf, W, H, focal_x, focal_y);
        P = make_float3(depth_cam * r.x, depth_cam * r.y, depth_cam);
    }

    const float3 light_pos = make_float3(0.0f, 0.0f, 0.0f);

    float3 L = normalize_or_default(
        make_float3(light_pos.x - P.x, light_pos.y - P.y, light_pos.z - P.z),
        make_float3(0.0f, 0.0f, -1.0f)
    );

    float3 V = normalize_or_default(
        make_float3(-view_ray.x, -view_ray.y, -view_ray.z),
        make_float3(0.0f, 0.0f, -1.0f)
    );

    float ndotv = n.x * V.x + n.y * V.y + n.z * V.z;
    if (ndotv < 0.0f)
    {
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
        ndotv = -ndotv;
    }

    float ndotl = n.x * L.x + n.y * L.y + n.z * L.z;

    o.ndotv = ndotv;
    o.ndotl = ndotl;

#if LIGHT_USE_LAMBERT
    float lambert = fmaxf(ndotl, 0.0f);
#else
    float lambert = 1.0f;
#endif

    o.lambert = lambert;
    o.spot = spotlight_factor(view_ray);

    float a = ambient_value(ambients);
    o.ambient = a;

    float dI_dummy = 0.0f;
    float I = intensity_value(intensity, &dI_dummy);
    o.I = I;
    o.dI_raw = dI_dummy;

    float3 LP = make_float3(P.x - light_pos.x, P.y - light_pos.y, P.z - light_pos.z);
    float dist2 = fmaxf(LP.x * LP.x + LP.y * LP.y + LP.z * LP.z, 1e-4f);

#if (FALLOFF_MODE == 1)
    o.inv = 1.0f / (1.0f + FALLOFF_K * dist2);
#else
    o.inv = 1.0f;
#endif

    o.Li = I * o.inv;
    #if (LIGHT_LI_CLAMP > 0)
        if (o.Li > (float)LIGHT_LI_CLAMP) {
            o.Li = (float)LIGHT_LI_CLAMP;
            o.li_clamped = 1.0f;
        }
    #endif
    o.intensity = o.Li;

    o.roughness = saturate01(rough_local);
    o.metallic = saturate01(metal_local);

    o.alpha = o.roughness * o.roughness;
    o.alpha2 = o.alpha * o.alpha;

    float3 base_clamped = make_float3(
        saturate01(base_color.x),
        saturate01(base_color.y),
        saturate01(base_color.z)
    );

    o.F0_rgb = make_float3(
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - o.metallic) + base_clamped.x * o.metallic,
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - o.metallic) + base_clamped.y * o.metallic,
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - o.metallic) + base_clamped.z * o.metallic
    );

    o.diffuse_brdf = 1.0f / LIGHT_PI;
    o.direct_diffuse_raw = o.diffuse_brdf * lambert * o.spot * o.Li;
    o.indirect_diffuse = a;

    float3 F_rgb = o.F0_rgb;
    float3 spec_brdf_rgb = make_float3(0.0f, 0.0f, 0.0f);

#if LIGHT_USE_PHONG
    float3 Hh = normalize_or_default(
        make_float3(L.x + V.x, L.y + V.y, L.z + V.z),
        make_float3(0.0f, 0.0f, -1.0f)
    );

    o.ndoth = fmaxf(n.x * Hh.x + n.y * Hh.y + n.z * Hh.z, 0.0f);
    o.vdoth = fmaxf(V.x * Hh.x + V.y * Hh.y + V.z * Hh.z, 0.0f);

    if (ndotl > 0.0f && ndotv > 0.0f)
    {
        float nv = fmaxf(ndotv, LIGHT_GGX_NV_EPS);
        float nl = fmaxf(ndotl, LIGHT_GGX_NL_EPS);

        o.D = ggx_D_and_dDdnh(o.ndoth, o.alpha2, nullptr);
        o.Gv = smith_G1_schlick_ggx(nv, o.roughness);
        o.Gl = smith_G1_schlick_ggx(nl, o.roughness);
        o.G = o.Gv * o.Gl;

        F_rgb = fresnel_schlick_rgb(o.vdoth, o.F0_rgb);
        float common = (o.D * o.G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);
        spec_brdf_rgb = float3_scale(F_rgb, common);
    }

    o.fresnel_rgb = F_rgb;
    o.fresnel = float3_avg(F_rgb);

    const float spec_nl = fmaxf(o.ndotl, 0.0f);
    o.spec_add_rgb = float3_scale(spec_brdf_rgb, spec_nl * o.spot * o.Li);
    o.spec_add = float3_avg(o.spec_add_rgb);
#endif

    float3 kd_rgb = make_float3(
        (1.0f - F_rgb.x) * (1.0f - o.metallic),
        (1.0f - F_rgb.y) * (1.0f - o.metallic),
        (1.0f - F_rgb.z) * (1.0f - o.metallic)
    );

    o.direct_diffuse_rgb = make_float3(
        o.direct_diffuse_raw * kd_rgb.x,
        o.direct_diffuse_raw * kd_rgb.y,
        o.direct_diffuse_raw * kd_rgb.z
    );

    o.indirect_approx_rgb = make_float3(
        o.indirect_diffuse * kd_rgb.x,
        o.indirect_diffuse * kd_rgb.y,
        o.indirect_diffuse * kd_rgb.z
    );

    o.diffuse_mul_rgb = make_float3(
        o.indirect_approx_rgb.x + o.direct_diffuse_rgb.x,
        o.indirect_approx_rgb.y + o.direct_diffuse_rgb.y,
        o.indirect_approx_rgb.z + o.direct_diffuse_rgb.z
    );

    o.diffuse_mul = float3_avg(o.diffuse_mul_rgb);
#endif

    return o;
}

__device__ __forceinline__
LightingOut eval_lighting(
    const float2& pixf,
    int W, int H,
    float focal_x, float focal_y,
    float3 normal_raw,
    float depth_cam,
    const float* __restrict__ ambients,
    const float* __restrict__ intensity,
    const float* __restrict__ roughness_raw,
    const float* __restrict__ metallic_raw,
    const float3& base_color,
    const float3* bu_cam_opt,
    const float3* bv_cam_opt,
    const float3* point_cam_opt = nullptr
) {
    LightingOut o = {};
    o.diffuse_mul        = 0.0f;
    o.diffuse_mul_rgb    = make_float3(0.0f, 0.0f, 0.0f);

    o.spec_add           = 0.0f;
    o.spec_add_rgb       = make_float3(0.0f, 0.0f, 0.0f);

    o.diffuse_rgb        = make_float3(0.0f, 0.0f, 0.0f);
    o.spec_rgb           = make_float3(0.0f, 0.0f, 0.0f);
    o.brdf_rgb           = make_float3(0.0f, 0.0f, 0.0f);
    o.indirect_approx_rgb = make_float3(0.0f, 0.0f, 0.0f);
    o.direct_diffuse_rgb  = make_float3(0.0f, 0.0f, 0.0f);

    o.diffuse_brdf       = 0.0f;

    o.indirect_diffuse   = 0.0f;
    o.direct_diffuse_raw = 0.0f;
    o.direct_diffuse     = 0.0f;

    // legacy-compatible aliases
    o.diffuse_amb        = 0.0f;
    o.diffuse_dir_raw    = 0.0f;
    o.diffuse_dir        = 0.0f;

    o.spec_pow           = 0.0f;
    o.spec_dir_raw       = 0.0f;
    o.spec_dir_gated     = 0.0f;
    o.spec_base          = 0.0f;

    o.spec_dir_raw_rgb   = make_float3(0.0f, 0.0f, 0.0f);
    o.spec_dir_gated_rgb = make_float3(0.0f, 0.0f, 0.0f);
    o.spec_base_rgb      = make_float3(0.0f, 0.0f, 0.0f);

    o.roughness          = LIGHT_GGX_ROUGHNESS;
    o.alpha              = LIGHT_GGX_ROUGHNESS * LIGHT_GGX_ROUGHNESS;
    o.alpha2             = o.alpha * o.alpha;

    o.F0                 = LIGHT_GGX_F0_DIELECTRIC;
    o.fresnel            = o.F0;
    o.D                  = 0.0f;
    o.G                  = 0.0f;
    o.Gv                 = 0.0f;
    o.Gl                 = 0.0f;

    o.lambert            = 1.0f;
    o.ndotl              = 1.0f;
    o.ndotv              = 1.0f;
    o.ndoth              = 0.0f;
    o.spot               = 1.0f;

    o.ambient            = 0.0f;
    o.intensity          = 1.0f;
    o.inv                = 1.0f;
    o.dintensity_ddepth  = 0.0f;
    o.li_clamped         = 0.0f;
    o.Li                 = 0.0f;
    o.I                  = 0.0f;

    o.dI_raw             = 0.0f;

    o.metallic           = LIGHT_GGX_METALLIC;
    o.dmetal_raw         = 0.0f;

    o.F0_rgb             = make_float3(LIGHT_GGX_F0_DIELECTRIC, LIGHT_GGX_F0_DIELECTRIC, LIGHT_GGX_F0_DIELECTRIC);
    o.fresnel_rgb        = o.F0_rgb;

#if (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
    float3 Ng = normalize_or_default(normal_raw, make_float3(0.f, 0.f, 1.f));
    float3 Ns = Ng;
    
    // ------------------------------------------------------------
    // Surface point / view direction source
    // ------------------------------------------------------------
    const bool use_point_cam = (point_cam_opt != nullptr);

    float3 P;        // point in camera space
    float3 view_ray; // camera -> surface direction, normalized

    if (use_point_cam)
    {
        P = *point_cam_opt;
        view_ray = normalize_or_default(P, make_float3(0.f, 0.f, 1.f));
    }
    else
    {
        view_ray = normalize_or_default(
            make_float3((pixf.x - 0.5f * W) / focal_x,
                        (pixf.y - 0.5f * H) / focal_y,
                        1.0f),
            make_float3(0.f, 0.f, 1.f));

        float3 r = compute_ray_unnorm(pixf, W, H, focal_x, focal_y);
        P = make_float3(depth_cam * r.x,
                        depth_cam * r.y,
                        depth_cam);
    }

    // ------------------------------------------------------------
    // Light position in camera space
    // ------------------------------------------------------------
    const float3 light_pos = make_float3(0.0f, 0.0f, 0.0f);

    // surface -> light
    float3 L = make_float3(light_pos.x - P.x,
                           light_pos.y - P.y,
                           light_pos.z - P.z);
    L = normalize_or_default(L, make_float3(0.f, 0.f, -1.f));

    // surface -> camera
    float3 V = make_float3(-view_ray.x, -view_ray.y, -view_ray.z);
    V = normalize_or_default(V, make_float3(0.f, 0.f, -1.f));

    float NgdotL = Ng.x * L.x + Ng.y * L.y + Ng.z * L.z;
    float NgdotV = Ng.x * V.x + Ng.y * V.y + Ng.z * V.z;

    #if LIGHT_USE_GEOMETRIC_HEMISPHERE_TEST
        if (NgdotL <= 0.0f || NgdotV <= 0.0f)
        {
            o.ndotl = 0.0f;
            o.ndotv = 0.0f;
            o.lambert = 0.0f;
            return o;
        }
    #endif

    // spotlight cone depends on camera -> surface direction
    float spot = spotlight_factor(view_ray);

    if (bu_cam_opt && bv_cam_opt)
    {
        float3 n_from_basis = cross(*bu_cam_opt, *bv_cam_opt);
        n_from_basis = normalize_or_default(n_from_basis, Ng);

         #if LIGHT_FACEFORWARD_SHADING_NORMAL
            float d = n_from_basis.x * Ng.x + n_from_basis.y * Ng.y + n_from_basis.z * Ng.z;
            if (d < 0.0f)
            {
                n_from_basis.x = -n_from_basis.x;
                n_from_basis.y = -n_from_basis.y;
                n_from_basis.z = -n_from_basis.z;
            }
        #endif

        #if LIGHT_USE_SHADING_NORMAL
            Ns = n_from_basis;
        #endif
    }
    
    float NsdotL = Ns.x * L.x + Ns.y * L.y + Ns.z * L.z;
    float NsdotV = Ns.x * V.x + Ns.y * V.y + Ns.z * V.z;

    o.ndotv = NsdotV;
    o.spot = spot;
    o.ndotl = NsdotL;

    // lambert
#if LIGHT_USE_LAMBERT
  #if LIGHT_LAMBERT_ABS
    float lambert = fabsf(o.ndotl);
  #else
    float lambert = fmaxf(o.ndotl, 0.0f);
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
    o.I = I;
    o.dI_raw = dI_draw; // dI/d(raw) if learnable, else 0

    // ------------------------------------------------------------
    // Distance / falloff
    // ------------------------------------------------------------
    float inv = 1.0f;
    o.dintensity_ddepth = 0.0f;

    float3 LP = make_float3(P.x - light_pos.x,
                            P.y - light_pos.y,
                            P.z - light_pos.z);

    float dist2 = fmaxf(LP.x * LP.x + LP.y * LP.y + LP.z * LP.z, 1e-4f);

#if (FALLOFF_MODE == 0)
    inv = 1.0f;
    o.dintensity_ddepth = 0.0f;

#elif (FALLOFF_MODE == 1)
    const float k = FALLOFF_K;
    inv = 1.0f / (1.0f + k * dist2);

    if (!use_point_cam)
    {
        // old depth-based derivative path
        float3 r = compute_ray_unnorm(pixf, W, H, focal_x, focal_y);
        float d_dist2_ddepth = 2.0f * (LP.x * r.x + LP.y * r.y + LP.z * 1.0f);
        float d_inv_d_dist2  = -k * inv * inv;
        o.dintensity_ddepth  = I * d_inv_d_dist2 * d_dist2_ddepth;
    }
    else
    {
        // Option C: no depth-based falloff gradient
        o.dintensity_ddepth = 0.0f;
    }
#else
    inv = 1.0f;
    o.dintensity_ddepth = 0.0f;
#endif

    o.inv = inv;
    o.Li_raw = I * inv;

    float Li = o.Li_raw;

#if (LIGHT_LI_CLAMP > 0)
    if (Li > (float)LIGHT_LI_CLAMP) {
        Li = (float)LIGHT_LI_CLAMP;
        o.li_clamped = 1.0f;
    }
#endif

    o.Li = Li;
    o.intensity = Li;

// metallic / roughness parameters
#if LIGHT_USE_PHONG
    float drough_draw = 0.0f;
    float rough = roughness_value(roughness_raw, &drough_draw);

    float dmetal_draw = 0.0f;
    float metallic = metallic_value(metallic_raw, &dmetal_draw);

    o.roughness = rough;
    o.drough_raw = drough_draw;

    o.alpha = rough * rough;
    o.alpha2 = o.alpha * o.alpha;

    o.metallic = metallic;
    o.dmetal_raw = dmetal_draw;

    // Make color stay in [0, 1] so F0 stays valid
    float3 base_clamped = make_float3(
        saturate01(base_color.x),
        saturate01(base_color.y),
        saturate01(base_color.z)
    );

    o.F0_rgb = make_float3(
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - metallic) + base_clamped.x * metallic,
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - metallic) + base_clamped.y * metallic,
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - metallic) + base_clamped.z * metallic
    );

    o.F0 = float3_avg(o.F0_rgb);
#endif

// diffuse decomposition
#if LIGHT_USE_LAMBERT
    o.diffuse_brdf = 1.0f / LIGHT_PI;

    // For now, before Fresnel is evaluated below, use dielectric fallback.
    // We will finalize diffuse_mul after the spec/Fresnel block.
    o.direct_diffuse_raw = o.diffuse_brdf * lambert * spot * Li;
    o.direct_diffuse     = o.direct_diffuse_raw;

    #if (LIGHT_AMBIENT_MODE != 0)
        o.indirect_diffuse = a;
    #else
        o.indirect_diffuse = 0.0f;
    #endif

    o.diffuse_mul = o.indirect_diffuse + o.direct_diffuse;
    o.diffuse_mul_rgb = make_float3(o.diffuse_mul, o.diffuse_mul, o.diffuse_mul);

    o.diffuse_amb     = o.indirect_diffuse;
    o.diffuse_dir_raw = o.direct_diffuse_raw;
    o.diffuse_dir     = o.direct_diffuse;

#else
    o.diffuse_brdf     = 0.0f;
    o.indirect_diffuse = 1.0f;
    o.direct_diffuse_raw = 0.0f;
    o.direct_diffuse   = 0.0f;
    o.diffuse_mul      = 1.0f;

    o.diffuse_amb      = o.indirect_diffuse;
    o.diffuse_dir_raw  = o.direct_diffuse_raw;
    o.diffuse_dir      = o.direct_diffuse;
#endif

#if LIGHT_USE_PHONG
    float3 Hh = normalize_or_default(
        make_float3(L.x + V.x, L.y + V.y, L.z + V.z),
        make_float3(0.f, 0.f, -1.f)
    );

    float ndoth = fmaxf(Ns.x * Hh.x + Ns.y * Hh.y + Ns.z * Hh.z, 0.0f);
    float vdoth = fmaxf(V.x * Hh.x + V.y * Hh.y + V.z * Hh.z, 0.0f);
    o.ndoth = ndoth;
    o.vdoth = vdoth;

    float D = 0.0f, G = 0.0f, Gv = 0.0f, Gl = 0.0f;
    float spec_brdf_scalar = 0.0f;
    float3 F_rgb = o.F0_rgb;
    float3 spec_brdf_rgb = make_float3(0.0f, 0.0f, 0.0f);

    if (o.ndotl > 0.0f && o.ndotv >= 0.0f)
    {
        float nv = fmaxf(o.ndotv, LIGHT_GGX_NV_EPS);
        float nl = fmaxf(o.ndotl, LIGHT_GGX_NL_EPS);

        D = ggx_D_and_dDdnh(ndoth, o.alpha2, nullptr);
        Gv = smith_G1_schlick_ggx(nv, o.roughness);
        Gl = smith_G1_schlick_ggx(nl, o.roughness);
        G  = Gv * Gl;

        F_rgb = fresnel_schlick_rgb(vdoth, o.F0_rgb);
        float common = (D * G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);

        spec_brdf_rgb = float3_scale(F_rgb, common);
        spec_brdf_scalar = float3_avg(spec_brdf_rgb);
    }

    const float spec_nl = fmaxf(o.ndotl, 0.0f);
    float3 spec_dir_raw_rgb = float3_scale(spec_brdf_rgb, spec_nl * spot * Li);
    float3 spec_dir_gated_rgb = spec_dir_raw_rgb;

    #if (LIGHT_SPEC_GATING == 1)
        if (o.ndotl <= 0.0f || o.ndotv <= 0.0f)
            spec_dir_gated_rgb = make_float3(0.0f, 0.0f, 0.0f);
    #endif

    o.D           = D;
    o.G           = G;
    o.Gv          = Gv;
    o.Gl          = Gl;
    o.fresnel_rgb = F_rgb;
    o.fresnel     = float3_avg(F_rgb);

    o.spec_pow        = D;
    o.spec_dir_raw    = float3_avg(spec_dir_raw_rgb);
    o.spec_dir_gated  = float3_avg(spec_dir_gated_rgb);
    o.spec_base       = o.spec_dir_gated;

    o.spec_dir_raw_rgb   = spec_dir_raw_rgb;
    o.spec_dir_gated_rgb = spec_dir_gated_rgb;
    o.spec_base_rgb      = spec_dir_gated_rgb;
    o.spec_add_rgb       = spec_dir_gated_rgb;

    // legacy scalar proxy for debug / compatibility
    o.spec_add        = float3_avg(o.spec_add_rgb);

    // finalize metallic-aware diffuse:
    // kD = (1 - F) * (1 - metallic)
    {
        const float3 kd_rgb = make_float3(
            (1.0f - F_rgb.x) * (1.0f - o.metallic),
            (1.0f - F_rgb.y) * (1.0f - o.metallic),
            (1.0f - F_rgb.z) * (1.0f - o.metallic)
        );

        // ---------------- material-only BRDF ----------------
        // diffuse_rgb/spec_rgb/brdf_rgb intentionally exclude ambient.
        o.diffuse_rgb = make_float3(
            o.diffuse_brdf * kd_rgb.x,
            o.diffuse_brdf * kd_rgb.y,
            o.diffuse_brdf * kd_rgb.z
        );

        o.spec_rgb = spec_brdf_rgb;

        o.brdf_rgb = make_float3(
            o.diffuse_rgb.x + o.spec_rgb.x,
            o.diffuse_rgb.y + o.spec_rgb.y,
            o.diffuse_rgb.z + o.spec_rgb.z
        );

        // ---------------- renderer shading decomposition ----------------
        // This is NOT part of the BRDF:
        // ambient is only an approximate indirect irradiance scalar.
        float3 indirect_diffuse_rgb = make_float3(
            o.indirect_diffuse * kd_rgb.x,
            o.indirect_diffuse * kd_rgb.y,
            o.indirect_diffuse * kd_rgb.z
        );

        o.indirect_approx_rgb = indirect_diffuse_rgb;

        o.direct_diffuse_rgb = make_float3(
            o.direct_diffuse_raw * kd_rgb.x,
            o.direct_diffuse_raw * kd_rgb.y,
            o.direct_diffuse_raw * kd_rgb.z
        );

        // Final diffuse shading multiplier used on base color:
        // base_color * (indirect_approx + direct_diffuse)
        o.diffuse_mul_rgb = make_float3(
            indirect_diffuse_rgb.x + o.direct_diffuse_rgb.x,
            indirect_diffuse_rgb.y + o.direct_diffuse_rgb.y,
            indirect_diffuse_rgb.z + o.direct_diffuse_rgb.z
        );

        // legacy scalar proxies kept for older paths / debug
        o.direct_diffuse = float3_avg(o.direct_diffuse_rgb);
        o.diffuse_amb    = o.indirect_diffuse;
        o.diffuse_dir    = o.direct_diffuse;
        o.diffuse_mul    = float3_avg(o.diffuse_mul_rgb);
    }
#else
    o.spec_pow          = 0.0f;
    o.spec_dir_raw      = 0.0f;
    o.spec_dir_gated    = 0.0f;
    o.spec_base         = 0.0f;
    o.spec_add          = 0.0f;
    o.spec_add_rgb      = make_float3(0.0f, 0.0f, 0.0f);

    o.spec_dir_raw_rgb   = make_float3(0.0f, 0.0f, 0.0f);
    o.spec_dir_gated_rgb = make_float3(0.0f, 0.0f, 0.0f);
    o.spec_base_rgb      = make_float3(0.0f, 0.0f, 0.0f);
#endif

#endif
    return o;
}