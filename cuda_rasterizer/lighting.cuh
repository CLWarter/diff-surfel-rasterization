#pragma once
#include "lighting_config.cuh"

struct LightingOut {
    // final grouped outputs
    float diffuse_mul;   // = diffuse_amb + diffuse_dir
    float spec_add;      // final additive specular

    // diffuse decomposition
    float diffuse_amb;       // additive ambient/base term
    float diffuse_dir_raw;   // lambert * spot * Li
    float diffuse_dir;       // directional diffuse after optional energy compensation

    // spec decomposition
    float spec_pow;         // ndoth^shiny
    float spec_dir_raw;      // spec_pow * spot * Li
    float spec_dir_gated;    // after LIGHT_SPEC_GATING, before ks
    float spec_base;         // alias/final pre-ks, old compatibility

    // angular/light terms
    float lambert;
    float ndotl;
    float ndotv;
    float ndoth;
    float spot;

    // ambient/intensity
    float ambient;
    float intensity;         // Li = I * inv * shade_conf (after clamp if used)
    float inv;
    float dintensity_ddepth;
    float Li;        // final intensity
    float Li_raw;    // I * inv
    float li_clamped;
    float I;

    // learnable parameter chain terms
    float dI_raw;
    float kspec;
    float dkspecular;
    float shiny;
    float dshin_raw;

    float shade_conf;
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
    float dL_ddiffuse,
    float dL_dspec)
{
    const float comp = 0.01f * 0.70710678f;
    const float3 light_pos = make_float3(-comp, -comp, 0.0f);

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
            float dLambert_from_diffuse = dL_ddiffuse * Lout.spot * Lout.intensity;
            #if LIGHT_ENERGY_COMP
                dLambert_from_diffuse *= (1.0f - Lout.kspec);
            #endif
            dL_dlambert += dLambert_from_diffuse;
        }
        #endif

        // spec contribution when spec is gated by lambert
        #if LIGHT_USE_PHONG && (LIGHT_SPEC_GATING == 2)
        {
            float dLambert_from_spec = dL_dspec * (Lout.kspec * Lout.spec_pow * Lout.spot * Lout.intensity);
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

    // -------- spec via ndoth wrt H(P) --------
    #if LIGHT_USE_PHONG
        {
            if (dL_dspec != 0.0f)
            {
                const float NDOTH_MAX = 0.95f;

                // forward used ndoth = clamp(dot(n_used, H), 0, NDOTH_MAX)
                const float ndoth = Lout.ndoth; // clamped forward value
                const float ndoth_raw = n_used.x * H.x + n_used.y * H.y + n_used.z * H.z;

                float dL_dndoth = 0.0f;

                // derivative exists only while ndoth_raw is inside unclamped interval
                if (ndoth_raw > 0.0f && ndoth_raw < NDOTH_MAX)
                {
                    float spec_deriv = 0.0f;
                    const float nd_eps = 1e-6f;

                    if (ndoth > nd_eps)
                        spec_deriv = Lout.shiny * (Lout.spec_pow / ndoth);

                    float dspec_dndoth = Lout.kspec * spec_deriv * Lout.spot * Lout.intensity;

                    #if (LIGHT_SPEC_GATING == 1)
                        if (Lout.ndotl <= 0.0f) dspec_dndoth = 0.0f;
                    #elif (LIGHT_SPEC_GATING == 2)
                        dspec_dndoth *= Lout.lambert;
                    #endif

                    dL_dndoth = dL_dspec * dspec_dndoth;
                }

                if (dL_dndoth != 0.0f)
                {
                    // ndoth = dot(n_used, H)
                    float3 gH = make_float3(
                        dL_dndoth * n_used.x,
                        dL_dndoth * n_used.y,
                        dL_dndoth * n_used.z
                    );

                    float3 gHraw = apply_norm_jacobian(Hraw, gH);

                    // Hraw = L + V
                    float3 gL = gHraw;
                    float3 gV = gHraw;

                    // V = normalize(-P)
                    float3 negP = make_float3(-P.x, -P.y, -P.z);
                    float3 gNegP = apply_norm_jacobian(negP, gV);
                    gP.x -= gNegP.x;
                    gP.y -= gNegP.y;
                    gP.z -= gNegP.z;

                    // L = normalize(light_pos - P)
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

            float dDiffuse_dSpot = 0.0f;
            #if LIGHT_USE_LAMBERT
                dDiffuse_dSpot = dL_ddiffuse * Lout.lambert * Lout.intensity;
                #if LIGHT_ENERGY_COMP
                    dDiffuse_dSpot *= (1.0f - Lout.kspec);
                #endif
            #endif

            float dSpec_dSpot = 0.0f;
            #if LIGHT_USE_PHONG
                float base = Lout.kspec * Lout.spec_pow * Lout.intensity;
                #if (LIGHT_SPEC_GATING == 1)
                    if (Lout.ndotl > 0.0f) dSpec_dSpot = dL_dspec * base;
                #elif (LIGHT_SPEC_GATING == 2)
                    dSpec_dSpot = dL_dspec * base * Lout.lambert;
                #else
                    dSpec_dSpot = dL_dspec * base;
                #endif
            #endif

            float dL_dcos = (dDiffuse_dSpot + dSpec_dSpot) * ds_dcos;

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
        float dDiff_dLi = 0.0f;
        #if LIGHT_USE_LAMBERT
            dDiff_dLi = Lout.lambert * Lout.spot;
            #if LIGHT_ENERGY_COMP
                dDiff_dLi *= (1.0f - Lout.kspec);
            #endif
        #endif

        float dSpec_dLi = 0.0f;
        #if LIGHT_USE_PHONG
            float base = Lout.kspec * Lout.spec_pow * Lout.spot;
            #if (LIGHT_SPEC_GATING == 1)
                if (Lout.ndotl <= 0.0f) base = 0.0f;
            #elif (LIGHT_SPEC_GATING == 2)
                base *= Lout.lambert;
            #endif
            dSpec_dLi = base;
        #endif

        float dL_dLi = dL_ddiffuse * dDiff_dLi + dL_dspec * dSpec_dLi;

        float3 LP = make_float3(P.x - light_pos.x, P.y - light_pos.y, P.z - light_pos.z);
        float dist2 = fmaxf(LP.x*LP.x + LP.y*LP.y + LP.z*LP.z, 1e-4f);

        const float k = FALLOFF_K;
        float inv = 1.0f / (1.0f + k * dist2);
        float dInv_dDist2 = -k * inv * inv;

        float dLi_dDist2 = Lout.I * Lout.shade_conf * dInv_dDist2;

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
    float3 n_raw,
    float depth_cam,
    const float* __restrict__ ambients,
    const float* __restrict__ intensity,
    const float* __restrict__ kspecs,
    const float* __restrict__ shiny,
    const float3* point_cam_opt = nullptr,
    float surface_conf = 1.0f,
    float alpha_local = 1.0f
) {
LightingOut o = {};
o.diffuse_mul      = 1.0f;
o.spec_add         = 0.0f;

o.diffuse_amb      = 0.0f;
o.diffuse_dir_raw  = 0.0f;
o.diffuse_dir      = 1.0f;

o.spec_pow        = 0.0f;
o.spec_dir_raw     = 0.0f;
o.spec_dir_gated   = 0.0f;
o.spec_base        = 0.0f;

o.lambert          = 1.0f;
o.ndotl            = 1.0f;
o.ndotv            = 1.0f;
o.ndoth            = 0.0f;
o.spot             = 1.0f;

o.ambient          = 0.0f;
o.intensity        = 1.0f;
o.inv              = 1.0f;
o.dintensity_ddepth = 0.0f;
o.li_clamped       = 0.0f;
o.Li = 0.0f;
o.Li_raw = 0.0f;
o.I = 0.0f;

o.dI_raw           = 0.0f;
o.kspec            = 0.0f;
o.dkspecular       = 0.0f;
o.shiny            = LIGHT_PHONG_SHININESS;
o.dshin_raw        = 0.0f;

o.shade_conf       = 1.0f;

#if (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
    float3 n = normalize_or_default(n_raw, make_float3(0.f, 0.f, 1.f));

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
    const float comp = 0.01f * 0.70710678f;
    const float3 light_pos = make_float3(-comp, -comp, 0.0f);

    // surface -> light
    float3 L = make_float3(light_pos.x - P.x,
                           light_pos.y - P.y,
                           light_pos.z - P.z);
    L = normalize_or_default(L, make_float3(0.f, 0.f, -1.f));

    // surface -> camera
    float3 V = make_float3(-view_ray.x, -view_ray.y, -view_ray.z);
    V = normalize_or_default(V, make_float3(0.f, 0.f, -1.f));

    // Stabilization
    float nv_raw = fmaxf(-(n.x * view_ray.x + n.y * view_ray.y + n.z * view_ray.z), 0.0f);

    float t = (nv_raw - LIGHT_NORMAL_GRAZING_END) /
              (LIGHT_NORMAL_GRAZING_START - LIGHT_NORMAL_GRAZING_END);
    t = fmaxf(0.0f, fminf(t, 1.0f));
    float grazing_t = 1.0f - smoothstep01(t);

    // extra stabilization for weak contributors
    float alpha_weak = 1.0f - saturate01(alpha_local / LIGHT_ALPHA_SOFT_REF);
    float alpha_boost = alpha_weak * LIGHT_NORMAL_ALPHA_BOOST;

    float3 n_view = make_float3(-view_ray.x, -view_ray.y, -view_ray.z);
    float blend_w = LIGHT_NORMAL_VIEW_BLEND * grazing_t + alpha_boost;
    blend_w = saturate01(blend_w);

    float3 n_stable = make_float3(
        (1.0f - blend_w) * n.x + blend_w * n_view.x,
        (1.0f - blend_w) * n.y + blend_w * n_view.y,
        (1.0f - blend_w) * n.z + blend_w * n_view.z
    );

    n = normalize_or_default(n_stable, make_float3(0.f, 0.f, 1.f)); 

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
    o.I = I;

float alpha_conf = saturate01(alpha_local / LIGHT_ALPHA_SOFT_REF);
float shade_conf = surface_conf * ((1.0f - LIGHT_WEAK_SHADE_REDUCTION) + LIGHT_WEAK_SHADE_REDUCTION * alpha_conf);

float Li = o.Li_raw * shade_conf;
#if (LIGHT_LI_CLAMP > 0)
    if (Li > (float)LIGHT_LI_CLAMP) {
        Li = (float)LIGHT_LI_CLAMP;
        o.li_clamped = 1.0f;
    }
#endif

o.shade_conf = shade_conf;
o.Li = Li;
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
    //dkspecular = fminf(dkspecular, 4.0f); // Clamp with 4.0 highest as more would be unrealistic (more for debug)

    float dshin_draw = 0.0f;
    float shin = shininess_value(shiny, &dshin_draw);
    //shin = fminf(shin, 64.0f); // Clamp with 64.0 highest as more would be unrealistic (mostly for debug)

    o.kspec = ks;
    o.dkspecular = dkspecular;
    o.shiny = shin;
    o.dshin_raw = dshin_draw;
#endif

// diffuse decomposition
#if LIGHT_USE_LAMBERT
    float diffuse_dir_raw = lambert * spot * Li;
    float diffuse_dir = diffuse_dir_raw;

    #if LIGHT_ENERGY_COMP && LIGHT_USE_PHONG
        diffuse_dir *= (1.0f - o.kspec);
    #endif

    float diffuse_amb = 0.0f;
    #if (LIGHT_AMBIENT_MODE != 0)
        diffuse_amb = a;
    #endif

    o.diffuse_dir_raw = diffuse_dir_raw;
    o.diffuse_dir     = diffuse_dir;
    o.diffuse_amb     = diffuse_amb;

    o.diffuse_mul     = diffuse_amb + diffuse_dir;
#else
    o.diffuse_dir_raw = 0.0f;
    o.diffuse_dir     = 0.0f;
    o.diffuse_amb     = 1.0f;
    o.diffuse_mul     = 1.0f;
#endif

    #if LIGHT_USE_PHONG
    float3 Hh = normalize_or_default(
        make_float3(L.x + V.x, L.y + V.y, L.z + V.z),
        make_float3(0.f, 0.f, -1.f)
    );

    const float NDOTH_MAX = 0.95f;
    float ndoth_raw = n.x * Hh.x + n.y * Hh.y + n.z * Hh.z;
    float ndoth = fminf(fmaxf(ndoth_raw, 0.0f), NDOTH_MAX);
    o.ndoth = ndoth;

    float spec_pow = (ndoth > 0.0f) ? powf(ndoth, o.shiny) : 0.0f;
    float spec_dir_raw = spec_pow * spot * Li;
    float spec_dir_gated = spec_dir_raw;

    #if (LIGHT_SPEC_GATING == 1)
        if (ndotl <= 0.0f) spec_dir_gated = 0.0f;
    #elif (LIGHT_SPEC_GATING == 2)
        spec_dir_gated *= lambert;
    #endif

    o.spec_pow      = spec_pow;
    o.spec_dir_raw   = spec_dir_raw;
    o.spec_dir_gated = spec_dir_gated;
    o.spec_base      = spec_dir_gated; //  old compatibility
    o.spec_add       = o.kspec * spec_dir_gated;
#else
    o.spec_pow      = 0.0f;
    o.spec_dir_raw   = 0.0f;
    o.spec_dir_gated = 0.0f;
    o.spec_base      = 0.0f;
    o.spec_add       = 0.0f;
#endif

#endif
    return o;
}