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

__device__ __forceinline__ float roughness_value(const float* roughness_raw, float* drough_draw_out)
{
#if (LIGHT_GGX_ROUGHNESS_MODE == 1)
    float t = sigmoidf_stable(roughness_raw[0]);
    if (drough_draw_out)
        *drough_draw_out = (1.0f - LIGHT_GGX_ROUGHNESS_MIN) * t * (1.0f - t);
    return LIGHT_GGX_ROUGHNESS_MIN + (1.0f - LIGHT_GGX_ROUGHNESS_MIN) * t;
#else
    (void)roughness_raw;
    if (drough_draw_out) *drough_draw_out = 0.0f;
    return fmaxf(LIGHT_GGX_ROUGHNESS, LIGHT_GGX_ROUGHNESS_MIN);
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

__device__ __forceinline__
float3 compute_ray_unnorm(const float2& pixf, int W, int H, float focal_x, float focal_y)
{
    float x = (pixf.x - 0.5f * (float)W) / focal_x;
    float y = (pixf.y - 0.5f * (float)H) / focal_y;
    return make_float3(x, y, 1.0f); // unnormalized
}

__device__ __forceinline__
float3 pointcam_lighting_grad_approx(
    const LightingOut& Lout,
    const float3& point_cam,
    const float3& n_used,      // use the same normal you use in backward approximation
    const float3& dL_ddiffuse_rgb,
    const float3& dL_dspec_rgb)
{
    const float3 light_pos = make_float3(0.0f, 0.0f, 0.0f);

    float3 P = point_cam;

    // view ray = normalize(P)
    float3 view_ray = normalize_or_default(P, make_float3(0.f, 0.f, 1.f));
    float3 V = normalize_or_default(make_float3(-view_ray.x, -view_ray.y, -view_ray.z),
                                    make_float3(0.f, 0.f, -1.f));

    // L = normalize(light_pos - P)
    float3 Lraw = make_float3(light_pos.x - P.x, light_pos.y - P.y, light_pos.z - P.z);
    float3 L = normalize_or_default(Lraw, make_float3(0.f, 0.f, -1.f));

    float3 Hraw = make_float3(L.x + V.x, L.y + V.y, L.z + V.z);
    float3 H = normalize_or_default(Hraw, make_float3(0.f, 0.f, -1.f));

    float3 gP = make_float3(0.f, 0.f, 0.f);

    // -------- diffuse via ndotl wrt L(P) --------
    // -------- diffuse + spec-through-lambert via ndotl wrt L(P) --------
    {
        float dL_dlambert = 0.0f;

        // diffuse contribution
        #if LIGHT_USE_LAMBERT
        {
            float dLambert_from_diffuse = 0.0f;
            if (Lout.lambert > 1e-6f)
            {
                dLambert_from_diffuse =
                    dL_ddiffuse_rgb.x * (Lout.direct_diffuse_rgb.x / Lout.lambert) +
                    dL_ddiffuse_rgb.y * (Lout.direct_diffuse_rgb.y / Lout.lambert) +
                    dL_ddiffuse_rgb.z * (Lout.direct_diffuse_rgb.z / Lout.lambert);
            }

            dL_dlambert += dLambert_from_diffuse;
        }
        #endif

        // spec contribution when spec is gated by lambert
        #if LIGHT_USE_PHONG && (LIGHT_SPEC_GATING == 2)
        {
            // spec_dir_raw already represents the current pre-gated scalar spec proxy.
            float dLambert_from_spec =
                dL_dspec_rgb.x * Lout.spec_dir_raw_rgb.x +
                dL_dspec_rgb.y * Lout.spec_dir_raw_rgb.y +
                dL_dspec_rgb.z * Lout.spec_dir_raw_rgb.z;
            dL_dlambert += dLambert_from_spec;
        }
        #endif

        float dL_dndotl = 0.0f;

        #if LIGHT_USE_LAMBERT_ABS
            if (Lout.ndotl > 0.0f) dL_dndotl = dL_dlambert;
            else if (Lout.ndotl < 0.0f) dL_dndotl = -dL_dlambert;
        #else
            if (Lout.ndotl > 0.0f) dL_dndotl = dL_dlambert;
        #endif

        if (dL_dndotl != 0.0f)
        {
            // ndotl = dot(n_used, L)
            // d(ndotl)/dL = n_used
            float3 gL = make_float3(
                dL_dndotl * n_used.x,
                dL_dndotl * n_used.y,
                dL_dndotl * n_used.z
            );

            // L = normalize(light_pos - P), so dL/dP = -J_norm(Lraw)
            float3 gLraw = apply_norm_jacobian(Lraw, gL);

            gP.x -= gLraw.x;
            gP.y -= gLraw.y;
            gP.z -= gLraw.z;
        }
    }

// -------- GGX RGB spec via ndoth / ndotl wrt point_cam --------
    #if LIGHT_USE_PHONG
        {
            if ((dL_dspec_rgb.x != 0.0f || dL_dspec_rgb.y != 0.0f || dL_dspec_rgb.z != 0.0f) &&
                Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
            {
                const float nh = fmaxf(Lout.ndoth, 1e-6f);
                const float a2 = Lout.alpha2;
                const float t = nh * nh * (a2 - 1.0f) + 1.0f;
                const float dD_dnh =
                    (-4.0f * LIGHT_PI * a2 * nh * (a2 - 1.0f) * t) /
                    fmaxf((LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS) * (LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS), 1e-8f);

                const float denom = fmaxf(
                    4.0f * fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS) * fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS),
                    LIGHT_GGX_DENOM_EPS
                );

                const float3 pref_rgb = make_float3(
                    (Lout.G * Lout.fresnel_rgb.x) / denom,
                    (Lout.G * Lout.fresnel_rgb.y) / denom,
                    (Lout.G * Lout.fresnel_rgb.z) / denom
                );

                float3 dspec_dndoth_rgb = make_float3(
                    dD_dnh * Lout.spot * Lout.Li * pref_rgb.x,
                    dD_dnh * Lout.spot * Lout.Li * pref_rgb.y,
                    dD_dnh * Lout.spot * Lout.Li * pref_rgb.z
                );

                const float common_nl = (4.0f * fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS)) / (denom * denom);

                float3 dspec_dndotl_rgb = make_float3(
                    -(Lout.D * Lout.G * Lout.fresnel_rgb.x) * common_nl * Lout.spot * Lout.Li,
                    -(Lout.D * Lout.G * Lout.fresnel_rgb.y) * common_nl * Lout.spot * Lout.Li,
                    -(Lout.D * Lout.G * Lout.fresnel_rgb.z) * common_nl * Lout.spot * Lout.Li
                );

                #if (LIGHT_SPEC_GATING == 1)
                    if (Lout.ndotl <= 0.0f) {
                        dspec_dndoth_rgb = make_float3(0.0f, 0.0f, 0.0f);
                        dspec_dndotl_rgb = make_float3(0.0f, 0.0f, 0.0f);
                    }
                #elif (LIGHT_SPEC_GATING == 2)
                    dspec_dndoth_rgb = make_float3(
                        dspec_dndoth_rgb.x * Lout.lambert,
                        dspec_dndoth_rgb.y * Lout.lambert,
                        dspec_dndoth_rgb.z * Lout.lambert
                    );
                    dspec_dndotl_rgb = make_float3(
                        dspec_dndotl_rgb.x + Lout.spec_dir_raw_rgb.x,
                        dspec_dndotl_rgb.y + Lout.spec_dir_raw_rgb.y,
                        dspec_dndotl_rgb.z + Lout.spec_dir_raw_rgb.z
                    );
                #endif

                float dL_dndoth =
                    dL_dspec_rgb.x * dspec_dndoth_rgb.x +
                    dL_dspec_rgb.y * dspec_dndoth_rgb.y +
                    dL_dspec_rgb.z * dspec_dndoth_rgb.z;

                float dL_dndotl =
                    dL_dspec_rgb.x * dspec_dndotl_rgb.x +
                    dL_dspec_rgb.y * dspec_dndotl_rgb.y +
                    dL_dspec_rgb.z * dspec_dndotl_rgb.z;

                if (dL_dndoth != 0.0f)
                {
                    float3 gH = make_float3(
                        dL_dndoth * n_used.x,
                        dL_dndoth * n_used.y,
                        dL_dndoth * n_used.z
                    );

                    float3 gHraw = apply_norm_jacobian(Hraw, gH);

                    float3 gL = gHraw;
                    float3 gV = gHraw;

                    float3 negP = make_float3(-P.x, -P.y, -P.z);
                    float3 gNegP = apply_norm_jacobian(negP, gV);
                    gP.x -= gNegP.x;
                    gP.y -= gNegP.y;
                    gP.z -= gNegP.z;

                    float3 gLraw = apply_norm_jacobian(Lraw, gL);
                    gP.x -= gLraw.x;
                    gP.y -= gLraw.y;
                    gP.z -= gLraw.z;
                }

                if (dL_dndotl != 0.0f)
                {
                    float3 gL = make_float3(
                        dL_dndotl * n_used.x,
                        dL_dndotl * n_used.y,
                        dL_dndotl * n_used.z
                    );

                    float3 gLraw = apply_norm_jacobian(Lraw, gL);
                    gP.x -= gLraw.x;
                    gP.y -= gLraw.y;
                    gP.z -= gLraw.z;
                }
            }
        }
    #endif

// -------- spotlight wrt point_cam --------
    #if LIGHT_USE_SPOT
    {
        float cosTheta = view_ray.z; // dot(view_ray, axis)

        const float innerCos = cosf(LIGHT_SPOT_INNER_DEG * (LIGHT_PI / 180.f));
        const float outerCos = cosf(LIGHT_SPOT_OUTER_DEG * (LIGHT_PI / 180.f));
        float denom = fmaxf(innerCos - outerCos, 1e-6f);

        float t = (cosTheta - outerCos) / denom;

        if (t > 0.0f && t < 1.0f)
        {
            float ds_dt = 6.0f * t * (1.0f - t);
            float ds_dcos = ds_dt / denom;

            if (LIGHT_SPOT_EXP != 1.0f)
            {
                float s0 = smoothstep01(t);
                ds_dcos *= LIGHT_SPOT_EXP * powf(fmaxf(s0, 1e-8f), LIGHT_SPOT_EXP - 1.0f);
            }

            float dL_dSpot = 0.0f;

            #if LIGHT_USE_LAMBERT
            {
                float3 dDiff_dSpot_rgb = make_float3(0.0f, 0.0f, 0.0f);
                if (Lout.spot > 1e-6f)
                {
                    dDiff_dSpot_rgb = make_float3(
                        Lout.direct_diffuse_rgb.x / Lout.spot,
                        Lout.direct_diffuse_rgb.y / Lout.spot,
                        Lout.direct_diffuse_rgb.z / Lout.spot
                    );
                }

                dL_dSpot += dL_ddiffuse_rgb.x * dDiff_dSpot_rgb.x;
                dL_dSpot += dL_ddiffuse_rgb.y * dDiff_dSpot_rgb.y;
                dL_dSpot += dL_ddiffuse_rgb.z * dDiff_dSpot_rgb.z;
            }
            #endif

            #if LIGHT_USE_PHONG
            {
                float3 dSpec_dSpot_rgb = make_float3(
                    Lout.spec_add_rgb.x / fmaxf(Lout.spot, 1e-6f),
                    Lout.spec_add_rgb.y / fmaxf(Lout.spot, 1e-6f),
                    Lout.spec_add_rgb.z / fmaxf(Lout.spot, 1e-6f)
                );

                #if (LIGHT_SPEC_GATING == 1)
                    if (Lout.ndotl <= 0.0f)
                        dSpec_dSpot_rgb = make_float3(0.0f, 0.0f, 0.0f);
                #endif

                dL_dSpot += dL_dspec_rgb.x * dSpec_dSpot_rgb.x;
                dL_dSpot += dL_dspec_rgb.y * dSpec_dSpot_rgb.y;
                dL_dSpot += dL_dspec_rgb.z * dSpec_dSpot_rgb.z;
            }
            #endif

            float dL_dcos = dL_dSpot * ds_dcos;

            float3 g_view = make_float3(0.f, 0.f, dL_dcos);
            float3 gP_view = apply_norm_jacobian(P, g_view);

            gP.x += gP_view.x;
            gP.y += gP_view.y;
            gP.z += gP_view.z;
        }
    }
    #endif

    // -------- falloff wrt point_cam --------
#if (FALLOFF_MODE == 1)
    {
        float dL_dLi = 0.0f;

        #if LIGHT_USE_LAMBERT
        {
            const float3 dDiff_dLi_rgb = make_float3(
                Lout.direct_diffuse_rgb.x / fmaxf(Lout.Li, 1e-6f),
                Lout.direct_diffuse_rgb.y / fmaxf(Lout.Li, 1e-6f),
                Lout.direct_diffuse_rgb.z / fmaxf(Lout.Li, 1e-6f)
            );

            dL_dLi += dL_ddiffuse_rgb.x * dDiff_dLi_rgb.x;
            dL_dLi += dL_ddiffuse_rgb.y * dDiff_dLi_rgb.y;
            dL_dLi += dL_ddiffuse_rgb.z * dDiff_dLi_rgb.z;
        }
        #endif

        #if LIGHT_USE_PHONG
        {
            float3 dSpec_dLi_rgb = make_float3(
                Lout.spec_add_rgb.x / fmaxf(Lout.Li, 1e-6f),
                Lout.spec_add_rgb.y / fmaxf(Lout.Li, 1e-6f),
                Lout.spec_add_rgb.z / fmaxf(Lout.Li, 1e-6f)
            );

            #if (LIGHT_SPEC_GATING == 1)
                if (Lout.ndotl <= 0.0f)
                    dSpec_dLi_rgb = make_float3(0.0f, 0.0f, 0.0f);
            #elif (LIGHT_SPEC_GATING == 2)
                dSpec_dLi_rgb = make_float3(
                    dSpec_dLi_rgb.x * Lout.lambert,
                    dSpec_dLi_rgb.y * Lout.lambert,
                    dSpec_dLi_rgb.z * Lout.lambert
                );
            #endif

            dL_dLi += dL_dspec_rgb.x * dSpec_dLi_rgb.x;
            dL_dLi += dL_dspec_rgb.y * dSpec_dLi_rgb.y;
            dL_dLi += dL_dspec_rgb.z * dSpec_dLi_rgb.z;
        }
        #endif

        float3 LP = make_float3(P.x - light_pos.x, P.y - light_pos.y, P.z - light_pos.z);
        float dist2 = fmaxf(LP.x*LP.x + LP.y*LP.y + LP.z*LP.z, 1e-4f);

        const float k = FALLOFF_K;
        float inv = 1.0f / (1.0f + k * dist2);
        float dInv_dDist2 = -k * inv * inv;

        float dLi_dDist2 = Lout.I * dInv_dDist2;

        #if (LIGHT_LI_CLAMP > 0)
            if (Lout.li_clamped > 0.5f)
                dLi_dDist2 = 0.0f;
        #endif

        float dL_dDist2 = dL_dLi * dLi_dDist2;

        gP.x += dL_dDist2 * 2.0f * LP.x;
        gP.y += dL_dDist2 * 2.0f * LP.y;
        gP.z += dL_dDist2 * 2.0f * LP.z;
    }
#endif

    return gP;
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
    float3 n = normalize_or_default(normal_raw, make_float3(0.f, 0.f, 1.f));

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

    float ndotv = n.x * V.x + n.y * V.y + n.z * V.z;
    o.ndotv = ndotv;

    // spotlight cone depends on camera -> surface direction
    float spot = spotlight_factor(view_ray);
    o.spot = spot;

    float ndotl = n.x * L.x + n.y * L.y + n.z * L.z;
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

    // scalar proxy stays for compatibility
    o.F0 = LIGHT_GGX_F0_DIELECTRIC * (1.0f - metallic) + float3_avg(base_color) * metallic;

    o.F0_rgb = make_float3(
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - metallic) + base_color.x * metallic,
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - metallic) + base_color.y * metallic,
        LIGHT_GGX_F0_DIELECTRIC * (1.0f - metallic) + base_color.z * metallic
    );
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

    float ndoth = fmaxf(n.x * Hh.x + n.y * Hh.y + n.z * Hh.z, 0.0f);
    float vdoth = fmaxf(V.x * Hh.x + V.y * Hh.y + V.z * Hh.z, 0.0f);
    o.ndoth = ndoth;
    o.vdoth = vdoth;

    float D = 0.0f, G = 0.0f, Gv = 0.0f, Gl = 0.0f;
    float spec_brdf_scalar = 0.0f;
    float3 F_rgb = o.F0_rgb;
    float3 spec_brdf_rgb = make_float3(0.0f, 0.0f, 0.0f);

    if (ndotl > 0.0f && o.ndotv > 0.0f)
    {
        float nv = fmaxf(o.ndotv, LIGHT_GGX_NV_EPS);
        float nl = fmaxf(ndotl, LIGHT_GGX_NL_EPS);

        D  = ggx_D(ndoth, o.alpha2);
        Gv = smith_G1_schlick_ggx(nv, o.roughness);
        Gl = smith_G1_schlick_ggx(nl, o.roughness);
        G  = Gv * Gl;

        F_rgb = fresnel_schlick_rgb(vdoth, o.F0_rgb);
        float common = (D * G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);

        spec_brdf_rgb = float3_scale(F_rgb, common);
        spec_brdf_scalar = float3_avg(spec_brdf_rgb);
    }

    float3 spec_dir_raw_rgb = float3_scale(spec_brdf_rgb, spot * Li);
    float3 spec_dir_gated_rgb = spec_dir_raw_rgb;

    #if (LIGHT_SPEC_GATING == 1)
        if (ndotl <= 0.0f) spec_dir_gated_rgb = make_float3(0.0f, 0.0f, 0.0f);
    #elif (LIGHT_SPEC_GATING == 2)
        spec_dir_gated_rgb = float3_scale(spec_dir_gated_rgb, lambert);
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
        o.indirect_approx_rgb = make_float3(
            o.indirect_diffuse,
            o.indirect_diffuse,
            o.indirect_diffuse
        );

        o.direct_diffuse_rgb = make_float3(
            o.direct_diffuse_raw * kd_rgb.x,
            o.direct_diffuse_raw * kd_rgb.y,
            o.direct_diffuse_raw * kd_rgb.z
        );

        // Final diffuse shading multiplier used on base color:
        // base_color * (indirect_approx + direct_diffuse)
        o.diffuse_mul_rgb = make_float3(
            o.indirect_approx_rgb.x + o.direct_diffuse_rgb.x,
            o.indirect_approx_rgb.y + o.direct_diffuse_rgb.y,
            o.indirect_approx_rgb.z + o.direct_diffuse_rgb.z
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