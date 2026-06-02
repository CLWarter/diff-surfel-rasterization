/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include "lighting.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

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

    // Scale by 1/|v|, the  Jacobian factor
    float3 result = make_float3(
        g_proj.x * inv_len,
        g_proj.y * inv_len,
        g_proj.z * inv_len
    );

    // extra clamp: prevents rare spikes even when len2 is barely above eps
    const float grad_max = 100.0f;
    result.x = fminf(fmaxf(result.x, -grad_max), grad_max);
    result.y = fminf(fmaxf(result.y, -grad_max), grad_max);
    result.z = fminf(fmaxf(result.z, -grad_max), grad_max);

    return result;
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
                    // d/d(ndotl) of [spec_brdf * lambert * spot * Li]
                    // = d(spec_brdf)/d(ndotl) * lambert 
                    // + spec_brdf * d(lambert)/d(ndotl) 
                    {
                        const float gating_factor = Lout.lambert * Lout.spot * Lout.Li;
                        float3 spec_pixel_contrib = make_float3(0.f, 0.f, 0.f);
                        
                        // If the gating factor is too small, zero out the scaling derivative
                        // component to avoid division spike jumps.
                        if (gating_factor > 1e-4f) {
                            spec_pixel_contrib = make_float3(
                                Lout.spec_dir_raw_rgb.x / gating_factor,
                                Lout.spec_dir_raw_rgb.y / gating_factor,
                                Lout.spec_dir_raw_rgb.z / gating_factor
                            );
                        }

                        dspec_dndotl_rgb = make_float3(
                            dspec_dndotl_rgb.x * Lout.lambert + spec_pixel_contrib.x,
                            dspec_dndotl_rgb.y * Lout.lambert + spec_pixel_contrib.y,
                            dspec_dndotl_rgb.z * Lout.lambert + spec_pixel_contrib.z
                        );
                        dspec_dndoth_rgb = make_float3(
                            dspec_dndoth_rgb.x * Lout.lambert,
                            dspec_dndoth_rgb.y * Lout.lambert,
                            dspec_dndoth_rgb.z * Lout.lambert
                        );
                    }
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


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ ambients,
	const float* __restrict__ intensity,
	const float* __restrict__ roughness,
	const float* __restrict__ metallic,
	const float* __restrict__ depths,
	const float3* __restrict__ means3D_cam,
	const float3* __restrict__ basis_u_cam,
    const float3* __restrict__ basis_v_cam,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dambients,
	float* __restrict__ dL_dintensity_raw,
	float* __restrict__ dL_droughness,
	float* __restrict__ dL_dmetallic,
	float3* __restrict__ dL_dbasis_u_cam,
	float3* __restrict__ dL_dbasis_v_cam)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	__shared__ float3 collected_center_cam[BLOCK_SIZE];
	__shared__ float3 collected_basis_u_cam[BLOCK_SIZE];
	__shared__ float3 collected_basis_v_cam[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float accum_eff[C] = {0};
	float last_eff[C]   = {0};
	float dL_dpixel[C];


#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}


	// per-thread accumulators
	float dAmb = 0.0f;
	float dSh = 0.0f;

#if LIGHT_SURFACE_SHADING_MODE

	float surf_w_sum = 0.0f;

	float3 surf_P_sum    = make_float3(0.0f, 0.0f, 0.0f);
	float3 surf_N_sum    = make_float3(0.0f, 0.0f, 0.0f);
	float3 surf_base_sum = make_float3(0.0f, 0.0f, 0.0f);

	float surf_rough_sum = 0.0f;
	float surf_metal_sum = 0.0f;
	float surf_depth_sum = 0.0f;

	float T_surface = 1.0f;

	for (int ii = 0, toDoSurf = range.y - range.x; ii < rounds; ii++, toDoSurf -= BLOCK_SIZE)
	{
		block.sync();

		int progress = ii * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];

			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];

			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};

			collected_center_cam[block.thread_rank()] = means3D_cam[coll_id];
			collected_basis_u_cam[block.thread_rank()] = basis_u_cam[coll_id];
			collected_basis_v_cam[block.thread_rank()] = basis_v_cam[coll_id];

			for (int cc = 0; cc < C; cc++)
				collected_colors[cc * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + cc];
		}

		block.sync();

		for (int jj = 0; inside && jj < min(BLOCK_SIZE, toDoSurf); jj++)
		{
			const float2 xy = collected_xy[jj];
			const float3 Tu = collected_Tu[jj];
			const float3 Tv = collected_Tv[jj];
			const float3 Tw = collected_Tw[jj];

			const float3 center_cam = collected_center_cam[jj];
			const float3 bu_cam = collected_basis_u_cam[jj];
			const float3 bv_cam = collected_basis_v_cam[jj];

			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (fabsf(p.z) < 1e-8f)
				continue;

			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = s.x * s.x + s.y * s.y;

			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

			const bool use_3d_footprint = (rho3d <= rho2d);
			float rho = min(rho3d, rho2d);

			float3 point_cam = make_float3(
				center_cam.x + s.x * bu_cam.x + s.y * bv_cam.x,
				center_cam.y + s.x * bu_cam.y + s.y * bv_cam.y,
				center_cam.z + s.x * bu_cam.z + s.y * bv_cam.z
			);

			float depth = s.x * Tw.x + s.y * Tw.y + Tw.z;
			bool depth_valid = true;

			if (!use_3d_footprint)
			{
				point_cam = center_cam;

	#if LIGHT_DEPTH_DISCARD_2D_FALLBACK
				depth_valid = false;
				depth = Tw.z;
	#else
				depth = Tw.z;
	#endif
			}

			if (depth_valid && depth < near_n)
				continue;

			float4 nor_o = collected_normal_opacity[jj];

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			float G = exp(power);
			float alpha = min(0.99f, nor_o.w * G);
			if (alpha < LIGHT_ALPHA_SKIP_THRESHOLD)
				continue;

			float w = alpha * T_surface;

			int gid = collected_id[jj];

			float3 base_rgb = make_float3(
				collected_colors[0 * BLOCK_SIZE + jj],
				collected_colors[1 * BLOCK_SIZE + jj],
				collected_colors[2 * BLOCK_SIZE + jj]
			);

			float3 n_basis = faceforward_basis_normal(bu_cam, bv_cam, point_cam);

			float3 n_stored = make_float3(nor_o.x, nor_o.y, nor_o.z);
			n_stored = normalize_or_default(n_stored, n_basis);

			float3 view_ray_ff = normalize_or_default(point_cam, make_float3(0.0f, 0.0f, 1.0f));
			float3 V_ff = make_float3(-view_ray_ff.x, -view_ray_ff.y, -view_ray_ff.z);

			float ndotv_stored =
				n_stored.x * V_ff.x +
				n_stored.y * V_ff.y +
				n_stored.z * V_ff.z;

			if (ndotv_stored < 0.0f)
			{
				n_stored.x = -n_stored.x;
				n_stored.y = -n_stored.y;
				n_stored.z = -n_stored.z;
			}

			const float normal_blend = 0.75f;

			float3 n_mix = make_float3(
				normal_blend * n_basis.x + (1.0f - normal_blend) * n_stored.x,
				normal_blend * n_basis.y + (1.0f - normal_blend) * n_stored.y,
				normal_blend * n_basis.z + (1.0f - normal_blend) * n_stored.z
			);

			n_mix = normalize_or_default(n_mix, n_basis);

			float ndiff =
				n_basis.x * n_stored.x +
				n_basis.y * n_stored.y +
				n_basis.z * n_stored.z;

			if (ndiff < 0.5f)
			{
				n_mix = n_stored;
			}

			float dmetal_dummy = 0.0f;
			float drough_dummy = 0.0f;

			float m_val = 0.0f;
			float r_val = 0.5f;

			#if (LIGHT_GGX_METALLIC_MODE == 1)
			if (metallic != nullptr)
				m_val = metallic_value(metallic + gid, &dmetal_dummy);
			#else
			m_val = metallic_value(nullptr, &dmetal_dummy);
			#endif

			#if (LIGHT_GGX_ROUGHNESS_MODE == 1)
			if (roughness != nullptr)
				r_val = roughness_value(roughness + gid, &drough_dummy);
			#else
			r_val = roughness_value(nullptr, &drough_dummy);
			#endif

			m_val = saturate01(m_val);
			r_val = saturate01(r_val);

			surf_w_sum += w;

			surf_P_sum.x += w * center_cam.x;
			surf_P_sum.y += w * center_cam.y;
			surf_P_sum.z += w * center_cam.z;

			surf_N_sum.x += w * n_mix.x;
			surf_N_sum.y += w * n_mix.y;
			surf_N_sum.z += w * n_mix.z;

			surf_base_sum.x += w * base_rgb.x;
			surf_base_sum.y += w * base_rgb.y;
			surf_base_sum.z += w * base_rgb.z;

			surf_rough_sum += w * r_val;
			surf_metal_sum += w * m_val;
			surf_depth_sum += w * center_cam.z;

			T_surface *= (1.0f - alpha);
			if (T_surface < 0.0001f)
				break;
		}
	}

	const float surface_alpha = 1.0f - T_surface;

	float3 surf_P = make_float3(0.0f, 0.0f, 0.0f);
	float3 surf_N = make_float3(0.0f, 0.0f, 1.0f);
	float3 surf_N_raw = make_float3(0.0f, 0.0f, 1.0f);
	float3 surf_base = make_float3(0.0f, 0.0f, 0.0f);

	float surf_rough = LIGHT_GGX_ROUGHNESS;
	float surf_metal = LIGHT_GGX_METALLIC;
	float surf_depth = 0.0f;

	if (surf_w_sum > 1e-8f)
	{
		float invW = 1.0f / surf_w_sum;

		surf_P = make_float3(
			surf_P_sum.x * invW,
			surf_P_sum.y * invW,
			surf_P_sum.z * invW
		);

		surf_N_raw = make_float3(
			surf_N_sum.x * invW,
			surf_N_sum.y * invW,
			surf_N_sum.z * invW
		);
		float3 surf_view_ray = normalize_or_default(surf_P, make_float3(0.0f, 0.0f, 1.0f));
		float3 surf_V = make_float3(-surf_view_ray.x, -surf_view_ray.y, -surf_view_ray.z);
		surf_N = normalize_or_default(surf_N_raw, make_float3(0.0f, 0.0f, 1.0f));
		{
			float ndotv_check = surf_N.x * surf_V.x + surf_N.y * surf_V.y + surf_N.z * surf_V.z;
			if (ndotv_check < 0.0f)
			{
				surf_N.x = -surf_N.x;
				surf_N.y = -surf_N.y;
				surf_N.z = -surf_N.z;
			}
		}

		surf_base = make_float3(
			surf_base_sum.x * invW,
			surf_base_sum.y * invW,
			surf_base_sum.z * invW
		);

		surf_rough = surf_rough_sum * invW;
		surf_metal = surf_metal_sum * invW;
		surf_depth = surf_depth_sum * invW;
	}

	LightingOut Lsurf = eval_lighting_surface_values(
		pixf, W, H, focal_x, focal_y,
		surf_N,
		surf_depth,
		ambients,
		intensity,
		surf_rough,
		surf_metal,
		surf_base,
		&surf_P
	);

	float3 dL_dsurf_base = make_float3(0.0f, 0.0f, 0.0f);
	float dL_dsurf_rough = 0.0f;
	float dL_dsurf_metal = 0.0f;
	// Exact P/N surface geometry derivatives are intentionally disabled here.
	// The surface-mode path below is exact for surface averaging, alpha/transmittance,
	// base color, roughness, metallic, ambient and intensity. Add analytic P/N later.
	float3 dL_dsurf_P = make_float3(0.0f, 0.0f, 0.0f);
	float3 dL_dsurf_N = make_float3(0.0f, 0.0f, 0.0f);
	float dL_dsurf_depth = 0.0f;

	float3 shaded = make_float3(
		surf_base.x * Lsurf.diffuse_mul_rgb.x + Lsurf.spec_add_rgb.x,
		surf_base.y * Lsurf.diffuse_mul_rgb.y + Lsurf.spec_add_rgb.y,
		surf_base.z * Lsurf.diffuse_mul_rgb.z + Lsurf.spec_add_rgb.z
	);

	float dL_dsurface_alpha = 0.0f;

	if (inside && surf_w_sum > 1e-8f)
	{
		dL_dsurface_alpha =
			dL_dpixel[0] * (shaded.x - bg_color[0]) +
			dL_dpixel[1] * (shaded.y - bg_color[1]) +
			dL_dpixel[2] * (shaded.z - bg_color[2]);

		// color = surface_alpha * (base * diffuse_mul + spec)
		dL_dsurf_base.x = surface_alpha * dL_dpixel[0] * Lsurf.diffuse_mul_rgb.x;
		dL_dsurf_base.y = surface_alpha * dL_dpixel[1] * Lsurf.diffuse_mul_rgb.y;
		dL_dsurf_base.z = surface_alpha * dL_dpixel[2] * Lsurf.diffuse_mul_rgb.z;

#if LIGHT_USE_PHONG
		if (Lsurf.ndotl > 0.0f && Lsurf.ndotv > 0.0f)
		{
			const float x = 1.0f - fmaxf(Lsurf.vdoth, 0.0f);
			const float x2 = x * x;
			const float x5 = x2 * x2 * x;
			const float dF_dF0 = 1.0f - x5;

			const float nv = fmaxf(Lsurf.ndotv, LIGHT_GGX_NV_EPS);
			const float nl = fmaxf(Lsurf.ndotl, LIGHT_GGX_NL_EPS);
			const float common = (Lsurf.D * Lsurf.G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);
			const float spec_scale = Lsurf.lambert * Lsurf.spot * Lsurf.Li;

			// F0_rgb = 0.04 * (1 - metallic) + base * metallic.
			// The specular-to-base gradient is intentionally blocked: we want the
			// stored base color to represent pure diffuse albedo, not to absorb
			// specular residuals. Roughness/metallic carry the specular signal.
		}

		// roughness gradient: D term + Smith G term
		if (Lsurf.ndotl > 0.0f && Lsurf.ndotv > 0.0f)
		{
			const float nh = fmaxf(Lsurf.ndoth, 1e-6f);
			const float a2 = fmaxf(Lsurf.alpha2, 1e-8f);
			const float r  = fmaxf(Lsurf.roughness, 1e-6f);

			const float t = nh * nh * (a2 - 1.0f) + 1.0f;
			const float denom = LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS;
			const float dD_da2 =
				(denom - a2 * (2.0f * LIGHT_PI * t * nh * nh)) /
				fmaxf(denom * denom, 1e-12f);

			const float nv = fmaxf(Lsurf.ndotv, LIGHT_GGX_NV_EPS);
			const float nl = fmaxf(Lsurf.ndotl, LIGHT_GGX_NL_EPS);
			const float invden = 1.0f / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);

			// D-path: dD/d(alpha) = dD_da2 * 2*alpha,  alpha = r^2
			const float alpha = r * r;
			const float dCommon_dalpha_D = dD_da2 * 2.0f * alpha * Lsurf.G * invden;

			// G-path: dG/d(alpha) = dG_dr * dr/d(alpha) = dG_dr / (2r)
			const float k_ggx = ((r + 1.0f) * (r + 1.0f)) * 0.125f;
			const float dk_dr = 0.25f * (r + 1.0f);

			const float denom_v = nv * (1.0f - k_ggx) + k_ggx + LIGHT_GGX_DENOM_EPS;
			const float denom_l = nl * (1.0f - k_ggx) + k_ggx + LIGHT_GGX_DENOM_EPS;

			const float ddenom_v_dr = (1.0f - nv) * dk_dr;
			const float ddenom_l_dr = (1.0f - nl) * dk_dr;

			const float dGv_dr = -nv * ddenom_v_dr / fmaxf(denom_v * denom_v, 1e-12f);
			const float dGl_dr = -nl * ddenom_l_dr / fmaxf(denom_l * denom_l, 1e-12f);
			const float dG_dr  = dGv_dr * Lsurf.Gl + Lsurf.Gv * dGl_dr;

			const float dr_dalpha      = 0.5f / r;
			const float dCommon_dalpha_G = Lsurf.D * (dG_dr * dr_dalpha) * invden;

			const float dCommon_dalpha = dCommon_dalpha_D + dCommon_dalpha_G;

			// dL_dsurf_rough = dL/d(alpha); multiplied by d(alpha)/d(raw) = drough_draw
			const float spec_scale = 4;
			const float brdf_scale = Lsurf.lambert * Lsurf.spot * Lsurf.Li;
			dL_dsurf_rough =
				surface_alpha * spec_scale * brdf_scale * (
					dL_dpixel[0] * Lsurf.fresnel_rgb.x * dCommon_dalpha +
					dL_dpixel[1] * Lsurf.fresnel_rgb.y * dCommon_dalpha +
					dL_dpixel[2] * Lsurf.fresnel_rgb.z * dCommon_dalpha
				);
		}

		// metallic gradient: diffuse kD path + specular F0 path
		{
			float dL_dm = 0.0f;

			const float x = 1.0f - fmaxf(Lsurf.vdoth, 0.0f);
			const float x2 = x * x;
			const float x5 = x2 * x2 * x;
			const float dF_dF0 = 1.0f - x5;

			const float3 dF0_dm = make_float3(
				saturate01(surf_base.x) - LIGHT_GGX_F0_DIELECTRIC,
				saturate01(surf_base.y) - LIGHT_GGX_F0_DIELECTRIC,
				saturate01(surf_base.z) - LIGHT_GGX_F0_DIELECTRIC
			);

			const float3 dF_dm = make_float3(
				dF_dF0 * dF0_dm.x,
				dF_dF0 * dF0_dm.y,
				dF_dF0 * dF0_dm.z
			);

			const float3 dkd_dm = make_float3(
				-(1.0f - Lsurf.fresnel_rgb.x) - (1.0f - Lsurf.metallic) * dF_dm.x,
				-(1.0f - Lsurf.fresnel_rgb.y) - (1.0f - Lsurf.metallic) * dF_dm.y,
				-(1.0f - Lsurf.fresnel_rgb.z) - (1.0f - Lsurf.metallic) * dF_dm.z
			);

			const float diffuse_light = Lsurf.indirect_diffuse + Lsurf.direct_diffuse_raw;

			dL_dm += surface_alpha * dL_dpixel[0] * surf_base.x * diffuse_light * dkd_dm.x;
			dL_dm += surface_alpha * dL_dpixel[1] * surf_base.y * diffuse_light * dkd_dm.y;
			dL_dm += surface_alpha * dL_dpixel[2] * surf_base.z * diffuse_light * dkd_dm.z;

			if (Lsurf.ndotl > 0.0f && Lsurf.ndotv > 0.0f)
			{
				const float nv = fmaxf(Lsurf.ndotv, LIGHT_GGX_NV_EPS);
				const float nl = fmaxf(Lsurf.ndotl, LIGHT_GGX_NL_EPS);
				const float common = (Lsurf.D * Lsurf.G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);
				const float spec_scale = Lsurf.lambert * Lsurf.spot * Lsurf.Li;

				dL_dm += surface_alpha * dL_dpixel[0] * dF0_dm.x * dF_dF0 * common * spec_scale;
				dL_dm += surface_alpha * dL_dpixel[1] * dF0_dm.y * dF_dF0 * common * spec_scale;
				dL_dm += surface_alpha * dL_dpixel[2] * dF0_dm.z * dF_dF0 * common * spec_scale;
			}

			dL_dsurf_metal = dL_dm;
		}
#endif

#if LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
{
    float3 kd_rgb = make_float3(
        (1.0f - Lsurf.fresnel_rgb.x) * (1.0f - surf_metal),
        (1.0f - Lsurf.fresnel_rgb.y) * (1.0f - surf_metal),
        (1.0f - Lsurf.fresnel_rgb.z) * (1.0f - surf_metal)
    );

    float dL_dambient =
        surface_alpha * (
            dL_dpixel[0] * surf_base.x * kd_rgb.x +
            dL_dpixel[1] * surf_base.y * kd_rgb.y +
            dL_dpixel[2] * surf_base.z * kd_rgb.z
        );

    float t_amb = sigmoidf_stable(ambients[0]);
    float da_draw = LIGHT_AMBIENT_MAX * t_amb * (1.0f - t_amb);

    dAmb += dL_dambient * da_draw;
}
#endif

#if (LIGHT_INTENSITY_MODE == 1)
		{
			float dL_dLi = 0.0f;

#if LIGHT_USE_LAMBERT
			dL_dLi += surface_alpha * dL_dpixel[0] * surf_base.x *
				(Lsurf.direct_diffuse_rgb.x / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[1] * surf_base.y *
				(Lsurf.direct_diffuse_rgb.y / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[2] * surf_base.z *
				(Lsurf.direct_diffuse_rgb.z / fmaxf(Lsurf.Li, 1e-6f));
#endif

#if LIGHT_USE_PHONG
			dL_dLi += surface_alpha * dL_dpixel[0] *
				(Lsurf.spec_add_rgb.x / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[1] *
				(Lsurf.spec_add_rgb.y / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[2] *
				(Lsurf.spec_add_rgb.z / fmaxf(Lsurf.Li, 1e-6f));
#endif

			float dL_dI = dL_dLi * Lsurf.inv;
			float dL_dIraw = dL_dI * Lsurf.dI_raw;

			#if (LIGHT_LI_CLAMP > 0)
				if (Lsurf.li_clamped > 0.5f)
					dL_dIraw = 0.0f;
			#endif

			atomicAdd(&dL_dintensity_raw[0], dL_dIraw);
		}
#endif
		// ---------- Surface point/depth gradient through falloff ----------
		{
			const float3 light_pos = make_float3(0.0f, 0.0f, 0.0f);

			float3 LP = make_float3(
				surf_P.x - light_pos.x,
				surf_P.y - light_pos.y,
				surf_P.z - light_pos.z
			);

			const float dist2 = fmaxf(
				LP.x * LP.x + LP.y * LP.y + LP.z * LP.z,
				1e-4f
			);

		#if (FALLOFF_MODE == 1)
			const float falloff_denom = 1.0f + FALLOFF_K * dist2;
			const float dinv_ddist2 =
				-FALLOFF_K / fmaxf(falloff_denom * falloff_denom, 1e-8f);
		#else
			const float dinv_ddist2 = 0.0f;
		#endif

			float dL_dLi = 0.0f;

		#if LIGHT_USE_LAMBERT
			dL_dLi += surface_alpha * dL_dpixel[0] * surf_base.x *
				(Lsurf.direct_diffuse_rgb.x / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[1] * surf_base.y *
				(Lsurf.direct_diffuse_rgb.y / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[2] * surf_base.z *
				(Lsurf.direct_diffuse_rgb.z / fmaxf(Lsurf.Li, 1e-6f));
		#endif

		#if LIGHT_USE_PHONG
			dL_dLi += surface_alpha * dL_dpixel[0] *
				(Lsurf.spec_add_rgb.x / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[1] *
				(Lsurf.spec_add_rgb.y / fmaxf(Lsurf.Li, 1e-6f));
			dL_dLi += surface_alpha * dL_dpixel[2] *
				(Lsurf.spec_add_rgb.z / fmaxf(Lsurf.Li, 1e-6f));
		#endif

			const float dL_dinv = dL_dLi * Lsurf.I;
			const float dL_ddist2 = dL_dinv * dinv_ddist2;

			dL_dsurf_P.x += dL_ddist2 * 2.0f * LP.x;
			dL_dsurf_P.y += dL_ddist2 * 2.0f * LP.y;
			dL_dsurf_P.z += dL_ddist2 * 2.0f * LP.z;

		}
		// ---------- Surface normal gradient through Lambert + GGX angular terms ----------
		{
			float3 gN = make_float3(0.0f, 0.0f, 0.0f);

			const float3 light_pos = make_float3(0.0f, 0.0f, 0.0f);

			float3 Lvec = normalize_or_default(
				make_float3(
					light_pos.x - surf_P.x,
					light_pos.y - surf_P.y,
					light_pos.z - surf_P.z
				),
				make_float3(0.0f, 0.0f, -1.0f)
			);

			float3 view_ray = normalize_or_default(
				surf_P,
				make_float3(0.0f, 0.0f, 1.0f)
			);

			float3 Vvec = normalize_or_default(
				make_float3(-view_ray.x, -view_ray.y, -view_ray.z),
				make_float3(0.0f, 0.0f, -1.0f)
			);

			float3 Hh = normalize_or_default(
				make_float3(
					Lvec.x + Vvec.x,
					Lvec.y + Vvec.y,
					Lvec.z + Vvec.z
				),
				make_float3(0.0f, 0.0f, -1.0f)
			);

			float dL_dndotl = 0.0f;

		#if LIGHT_USE_LAMBERT
			if (Lsurf.ndotl > 0.0f && Lsurf.lambert > 1e-6f)
			{
				dL_dndotl += surface_alpha * (
					dL_dpixel[0] * surf_base.x *
						(Lsurf.direct_diffuse_rgb.x / Lsurf.lambert) +
					dL_dpixel[1] * surf_base.y *
						(Lsurf.direct_diffuse_rgb.y / Lsurf.lambert) +
					dL_dpixel[2] * surf_base.z *
						(Lsurf.direct_diffuse_rgb.z / Lsurf.lambert)
				);
			}
		#endif

		#if LIGHT_USE_PHONG
			if (Lsurf.ndotl > 0.0f && Lsurf.ndotv > 0.0f)
			{
				const float nv = fmaxf(Lsurf.ndotv, LIGHT_GGX_NV_EPS);
				const float nl = fmaxf(Lsurf.ndotl, LIGHT_GGX_NL_EPS);
				const float nh = fmaxf(Lsurf.ndoth, 1e-6f);

				const float denom_raw = 4.0f * nv * nl;
				const float denom = fmaxf(denom_raw, LIGHT_GGX_DENOM_EPS);
				const float inv_denom = 1.0f / denom;

				float dD_dnh = 0.0f;
				ggx_D_and_dDdnh(Lsurf.ndoth, Lsurf.alpha2, &dD_dnh);

				const float r = fmaxf(Lsurf.roughness, 1e-6f);
				const float k = ((r + 1.0f) * (r + 1.0f)) * 0.125f;

				const float denom_v = nv * (1.0f - k) + k + LIGHT_GGX_DENOM_EPS;
				const float denom_l = nl * (1.0f - k) + k + LIGHT_GGX_DENOM_EPS;

				const float dGv_dnv =
					(k + LIGHT_GGX_DENOM_EPS) / fmaxf(denom_v * denom_v, 1e-12f);
				const float dGl_dnl =
					(k + LIGHT_GGX_DENOM_EPS) / fmaxf(denom_l * denom_l, 1e-12f);

				const float dG_dnv = Lsurf.Gl * dGv_dnv;
				const float dG_dnl = Lsurf.Gv * dGl_dnl;

				float dcommon_dnh = dD_dnh * Lsurf.G * inv_denom;
				float dcommon_dnv = 0.0f;
				float dcommon_dnl = 0.0f;

				if (denom_raw > LIGHT_GGX_DENOM_EPS)
				{
					const float inv_denom2 = inv_denom * inv_denom;

					dcommon_dnv =
						Lsurf.D * (dG_dnv * inv_denom - Lsurf.G * (4.0f * nl) * inv_denom2);

					dcommon_dnl =
						Lsurf.D * (dG_dnl * inv_denom - Lsurf.G * (4.0f * nv) * inv_denom2);
				}
				else
				{
					dcommon_dnv = Lsurf.D * dG_dnv * inv_denom;
					dcommon_dnl = Lsurf.D * dG_dnl * inv_denom;
				}

				const float spotLi = Lsurf.spot * Lsurf.Li;

				const float dL_dndoth =
					surface_alpha * (
						dL_dpixel[0] * Lsurf.fresnel_rgb.x * dcommon_dnh * spotLi +
						dL_dpixel[1] * Lsurf.fresnel_rgb.y * dcommon_dnh * spotLi +
						dL_dpixel[2] * Lsurf.fresnel_rgb.z * dcommon_dnh * spotLi
					);

				const float dL_dndotv =
					surface_alpha * (
						dL_dpixel[0] * Lsurf.fresnel_rgb.x * dcommon_dnv * spotLi +
						dL_dpixel[1] * Lsurf.fresnel_rgb.y * dcommon_dnv * spotLi +
						dL_dpixel[2] * Lsurf.fresnel_rgb.z * dcommon_dnv * spotLi
					);

				const float dL_dndotl_spec =
					surface_alpha * (
						dL_dpixel[0] * Lsurf.fresnel_rgb.x * dcommon_dnl * spotLi +
						dL_dpixel[1] * Lsurf.fresnel_rgb.y * dcommon_dnl * spotLi +
						dL_dpixel[2] * Lsurf.fresnel_rgb.z * dcommon_dnl * spotLi
					);

				gN.x += dL_dndoth * Hh.x + dL_dndotv * Vvec.x + dL_dndotl_spec * Lvec.x;
				gN.y += dL_dndoth * Hh.y + dL_dndotv * Vvec.y + dL_dndotl_spec * Lvec.y;
				gN.z += dL_dndoth * Hh.z + dL_dndotv * Vvec.z + dL_dndotl_spec * Lvec.z;
			}
		#endif

			gN.x += dL_dndotl * Lvec.x;
			gN.y += dL_dndotl * Lvec.y;
			gN.z += dL_dndotl * Lvec.z;

			dL_dsurf_N.x += gN.x;
			dL_dsurf_N.y += gN.y;
			dL_dsurf_N.z += gN.z;
		}
	}

	float3 dL_dsurf_N_raw = make_float3(0.0f, 0.0f, 0.0f);

	if (surf_w_sum > 1e-8f)
	{
		dL_dsurf_N_raw = apply_norm_jacobian(surf_N_raw, dL_dsurf_N);
	}

	float future_weight_grad = 0.0f;

#endif

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;


	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			collected_center_cam[block.thread_rank()] = means3D_cam[coll_id];
			collected_basis_u_cam[block.thread_rank()] = basis_u_cam[coll_id];
    		collected_basis_v_cam[block.thread_rank()] = basis_v_cam[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			const float3 center_cam = collected_center_cam[j];
			const float3 bu_cam = collected_basis_u_cam[j];
			const float3 bv_cam = collected_basis_v_cam[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (fabsf(p.z) < 1e-8f) continue;
						float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 

			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			const bool use_3d_footprint = (rho3d <= rho2d);
			float rho = min(rho3d, rho2d);

			// 3D surfel hit point in camera space.
			// Only reliable when the 3D footprint wins over the 2D low-pass fallback.
			float3 point_cam = make_float3(
				center_cam.x + s.x * bu_cam.x + s.y * bv_cam.x,
				center_cam.y + s.x * bu_cam.y + s.y * bv_cam.y,
				center_cam.z + s.x * bu_cam.z + s.y * bv_cam.z
			);

			// Per-pixel surfel depth: Tw * [u, v, 1].
			// If the 2D low-pass filter wins, fall back to center depth.
			float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			bool depth_valid = true;

			if (!use_3d_footprint)
			{
				point_cam = center_cam;

				#if LIGHT_DEPTH_DISCARD_2D_FALLBACK
					depth_valid = false;
					c_d = Tw.z; // keep harmless fallback for lighting/debug if needed
				#else
					c_d = Tw.z;
				#endif
			}

			if (depth_valid && c_d < near_n)
    			continue;
			
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < LIGHT_ALPHA_SKIP_THRESHOLD)
				continue;

			T = T / (1.f - alpha);

			// ================= LAMBERT + PHONG SHADING (BACKWARD) ======================

			float dL_dalpha = 0.0f;

			#if LIGHT_SURFACE_SHADING_MODE
			float dL_dalpha_surface = 0.0f;
			#endif

			#if RENDER_AXUTILITY
			float dL_dalpha_aux = 0.0f;
			#endif

			const int global_id = collected_id[j];

			// depth accumulator
            float dL_dz      = 0.0f;

			float extra_dL_dsx = 0.0f;
			float extra_dL_dsy = 0.0f;

			LightingOut Lout = {};
			const float* rough_ptr = nullptr;
			const float* metal_ptr = nullptr;

#if LIGHT_SURFACE_SHADING_MODE
{
    const float w = alpha * T;

    if (surf_w_sum > 1e-8f)
    {
        const float invW = 1.0f / surf_w_sum;
        const float one_minus_alpha = fmaxf(1.0f - alpha, 1e-6f);

        float dL_dw = 0.0f;

		const float safe_depth = depth_valid ? c_d : point_cam.z;

		const float coeff = w * invW;
		const int gid = collected_id[j];

		float3 base_rgb = make_float3(
			collected_colors[0 * BLOCK_SIZE + j],
			collected_colors[1 * BLOCK_SIZE + j],
			collected_colors[2 * BLOCK_SIZE + j]
		);
		
		float3 n_basis = faceforward_basis_normal(bu_cam, bv_cam, point_cam);

		float3 n_stored = make_float3(normal[0], normal[1], normal[2]);
		n_stored = normalize_or_default(n_stored, n_basis);

		// Faceforward stored normal too.
		float3 view_ray_ff = normalize_or_default(point_cam, make_float3(0.0f, 0.0f, 1.0f));
		float3 V_ff = make_float3(-view_ray_ff.x, -view_ray_ff.y, -view_ray_ff.z);

		float ndotv_stored =
			n_stored.x * V_ff.x +
			n_stored.y * V_ff.y +
			n_stored.z * V_ff.z;

		if (ndotv_stored < 0.0f)
		{
			n_stored.x = -n_stored.x;
			n_stored.y = -n_stored.y;
			n_stored.z = -n_stored.z;
		}

		const float normal_blend = 0.75f;

		float3 n_mix = make_float3(
			normal_blend * n_basis.x + (1.0f - normal_blend) * n_stored.x,
			normal_blend * n_basis.y + (1.0f - normal_blend) * n_stored.y,
			normal_blend * n_basis.z + (1.0f - normal_blend) * n_stored.z
		);

		n_mix = normalize_or_default(n_mix, n_basis);

		float ndiff =
			n_basis.x * n_stored.x +
			n_basis.y * n_stored.y +
			n_basis.z * n_stored.z;

		const bool use_stored_normal = (ndiff < 0.5f);

		if (use_stored_normal)
		{
			n_mix = n_stored;
		}

		 LightMaterialValues mat = eval_light_material_values(
			metallic != nullptr ? metallic + gid : nullptr,
			roughness != nullptr ? roughness + gid : nullptr
		);

		float m_val = mat.metallic;
		float r_val = mat.roughness;

		float dmetal_draw = mat.dmetal_draw;
		float drough_draw = mat.drough_draw;

		atomicAdd(&dL_dcolors[gid * C + 0], coeff * dL_dsurf_base.x);
		atomicAdd(&dL_dcolors[gid * C + 1], coeff * dL_dsurf_base.y);
		atomicAdd(&dL_dcolors[gid * C + 2], coeff * dL_dsurf_base.z);

#if (LIGHT_GGX_ROUGHNESS_MODE == 1)
		atomicAdd(&dL_droughness[gid], coeff * dL_dsurf_rough * 2.0f * Lsurf.roughness * drough_draw);
#endif

#if (LIGHT_GGX_METALLIC_MODE == 1)
		atomicAdd(&dL_dmetallic[gid], coeff * dL_dsurf_metal * dmetal_draw);
#endif

		dL_dw += dL_dsurf_base.x * (base_rgb.x - surf_base.x) * invW;
		dL_dw += dL_dsurf_base.y * (base_rgb.y - surf_base.y) * invW;
		dL_dw += dL_dsurf_base.z * (base_rgb.z - surf_base.z) * invW;

		dL_dw += dL_dsurf_rough * 2.0f * Lsurf.roughness * (r_val - surf_rough) * invW;
		dL_dw += dL_dsurf_metal * (m_val - surf_metal) * invW;

		dL_dw += dL_dsurf_P.x * (point_cam.x - surf_P.x) * invW;
		dL_dw += dL_dsurf_P.y * (point_cam.y - surf_P.y) * invW;
		dL_dw += dL_dsurf_P.z * (point_cam.z - surf_P.z) * invW;

		dL_dw += dL_dsurf_N_raw.x * (n_mix.x - surf_N_raw.x) * invW;
		dL_dw += dL_dsurf_N_raw.y * (n_mix.y - surf_N_raw.y) * invW;
		dL_dw += dL_dsurf_N_raw.z * (n_mix.z - surf_N_raw.z) * invW;

		const float coeff_norm = w * invW;

		float3 dL_dn_mix = make_float3(
			coeff_norm * dL_dsurf_N_raw.x,
			coeff_norm * dL_dsurf_N_raw.y,
			coeff_norm * dL_dsurf_N_raw.z
		);

		float3 dL_dbu = make_float3(0.0f, 0.0f, 0.0f);
		float3 dL_dbv = make_float3(0.0f, 0.0f, 0.0f);

		if (!use_stored_normal)
		{
			// n_mix = normalize(normal_blend * n_basis + (1-normal_blend) * n_stored)
			float3 n_pre = make_float3(
				normal_blend * n_basis.x + (1.0f - normal_blend) * n_stored.x,
				normal_blend * n_basis.y + (1.0f - normal_blend) * n_stored.y,
				normal_blend * n_basis.z + (1.0f - normal_blend) * n_stored.z
			);

			float3 dL_dn_pre = apply_norm_jacobian(n_pre, dL_dn_mix);

			float3 dL_dn_basis = make_float3(
				normal_blend * dL_dn_pre.x,
				normal_blend * dL_dn_pre.y,
				normal_blend * dL_dn_pre.z
			);

			float3 c_basis = cross(bu_cam, bv_cam);
			float3 dL_dc_basis = apply_norm_jacobian(c_basis, dL_dn_basis);

			dL_dbu = cross(bv_cam, dL_dc_basis);
			dL_dbv = cross(dL_dc_basis, bu_cam);
		}

		const float coeff_geom = w * invW;

		extra_dL_dsx += coeff_geom * (
			dL_dsurf_P.x * bu_cam.x +
			dL_dsurf_P.y * bu_cam.y +
			dL_dsurf_P.z * bu_cam.z
		);

		extra_dL_dsy += coeff_geom * (
			dL_dsurf_P.x * bv_cam.x +
			dL_dsurf_P.y * bv_cam.y +
			dL_dsurf_P.z * bv_cam.z
		);

		atomicAdd(&dL_dbasis_u_cam[global_id].x, dL_dbu.x);
		atomicAdd(&dL_dbasis_u_cam[global_id].y, dL_dbu.y);
		atomicAdd(&dL_dbasis_u_cam[global_id].z, dL_dbu.z);

		atomicAdd(&dL_dbasis_v_cam[global_id].x, dL_dbv.x);
		atomicAdd(&dL_dbasis_v_cam[global_id].y, dL_dbv.y);
		atomicAdd(&dL_dbasis_v_cam[global_id].z, dL_dbv.z);

		dL_dalpha_surface +=
			T * dL_dw
			- future_weight_grad / one_minus_alpha;

		future_weight_grad += dL_dw * w;

		dL_dalpha_surface +=
			dL_dsurface_alpha * T_surface / one_minus_alpha;
    }
}
#elif LIGHT_ENABLE_BWD && (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)

				// Evaluate lighting terms matching forward
				float3 n_raw = make_float3(normal[0], normal[1], normal[2]);

				// per gaussian pointers
				const int gid = collected_id[j];
				rough_ptr = roughness + gid;
				metal_ptr = metallic + gid;

				float3 base_rgb = make_float3(
					collected_colors[0 * BLOCK_SIZE + j],
					collected_colors[1 * BLOCK_SIZE + j],
					collected_colors[2 * BLOCK_SIZE + j]
				);

				Lout = eval_lighting(
					pixf, W, H, focal_x, focal_y,
					n_raw, c_d,
					ambients, intensity,
					rough_ptr, metal_ptr,
					base_rgb,
					&bu_cam,
    				&bv_cam,
					&point_cam
				);

				const float w = alpha * T;

				// Accumulators for lighting parameter gradients
				float3 dL_ddiffuse_rgb = make_float3(0.0f, 0.0f, 0.0f);
				float3 dL_dspec_rgb    = make_float3(0.0f, 0.0f, 0.0f);

				#if LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
					float dL_dindirect_approx = 0.0f; // accum for ambient gradient
				#endif

				for (int ch = 0; ch < C; ch++)
				{
					const float c = collected_colors[ch * BLOCK_SIZE + j];
					const float dL_dchannel = dL_dpixel[ch];

					float dchannel_dcolor = w * Lout.diffuse_mul;
					#if LIGHT_USE_PHONG
						if (ch == 0) dchannel_dcolor = w * Lout.diffuse_mul_rgb.x;
						else if (ch == 1) dchannel_dcolor = w * Lout.diffuse_mul_rgb.y;
						else if (ch == 2) dchannel_dcolor = w * Lout.diffuse_mul_rgb.z;
					#endif

					// ambient gradient
					#if LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
						// forward: channel_contribution = w * c * (indirect_diffuse * kd_rgb[ch] + ...)
						// so d/d(indirect_diffuse) = w * c * kd_rgb[ch]
						{
							float kd_ch = 1.0f;
							#if LIGHT_USE_PHONG
								if      (ch == 0) kd_ch = (1.0f - Lout.fresnel_rgb.x) * (1.0f - Lout.metallic);
								else if (ch == 1) kd_ch = (1.0f - Lout.fresnel_rgb.y) * (1.0f - Lout.metallic);
								else if (ch == 2) kd_ch = (1.0f - Lout.fresnel_rgb.z) * (1.0f - Lout.metallic);
							#endif
							dL_dindirect_approx += dL_dchannel * (w * c * kd_ch);
						}
					#endif

					// reccurence
					accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = c;

					float eff = c * Lout.diffuse_mul;
					#if LIGHT_USE_PHONG
						if (ch == 0) eff = c * Lout.diffuse_mul_rgb.x + Lout.spec_add_rgb.x;
						else if (ch == 1) eff = c * Lout.diffuse_mul_rgb.y + Lout.spec_add_rgb.y;
						else if (ch == 2) eff = c * Lout.diffuse_mul_rgb.z + Lout.spec_add_rgb.z;
					#endif

					accum_eff[ch] = last_alpha * last_eff[ch] + (1.f - last_alpha) * accum_eff[ch];
					last_eff[ch] = eff;

					// alpha influence contribution of gaussians to accumulated behind
					dL_dalpha += (eff - accum_eff[ch]) * dL_dchannel;

					// gradients wrt diffuse/spec RGB outputs
					#if LIGHT_USE_PHONG
						if (ch == 0)
						{
							dL_ddiffuse_rgb.x += dL_dchannel * (w * c);
							dL_dspec_rgb.x    += dL_dchannel * w;
						}
						else if (ch == 1)
						{
							dL_ddiffuse_rgb.y += dL_dchannel * (w * c);
							dL_dspec_rgb.y    += dL_dchannel * w;
						}
						else if (ch == 2)
						{
							dL_ddiffuse_rgb.z += dL_dchannel * (w * c);
							dL_dspec_rgb.z    += dL_dchannel * w;
						}
					#endif

					// base color gradient
					atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				}

				#if LIGHT_USE_PHONG
				if (Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
				{
					const float x = 1.0f - fmaxf(Lout.vdoth, 0.0f);
					const float x2 = x * x;
					const float x5 = x2 * x2 * x;
					const float dF_dF0 = 1.0f - x5;

					const float nv = fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS);
					const float nl = fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS);
					const float common = (Lout.D * Lout.G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);

					const float dF0_dbase = Lout.metallic;
					const float dspec_dbase = dF0_dbase * dF_dF0 * common * Lout.lambert * Lout.spot * Lout.Li;

					atomicAdd(&(dL_dcolors[global_id * C + 0]), dL_dspec_rgb.x * dspec_dbase);
					atomicAdd(&(dL_dcolors[global_id * C + 1]), dL_dspec_rgb.y * dspec_dbase);
					atomicAdd(&(dL_dcolors[global_id * C + 2]), dL_dspec_rgb.z * dspec_dbase);
				}
				#endif

			// -------- intensity (falloff) to depth gradient --------
			{
				float dL_dLi = 0.0f;

				#if LIGHT_USE_LAMBERT
				{
					float3 dDiff_dLi_rgb = make_float3(
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
					#endif

					dL_dLi += dL_dspec_rgb.x * dSpec_dLi_rgb.x;
					dL_dLi += dL_dspec_rgb.y * dSpec_dLi_rgb.y;
					dL_dLi += dL_dspec_rgb.z * dSpec_dLi_rgb.z;
				}
				#endif

				// If forward clamped Li, stop gradients through Li to avoid it fighting the clamp
				float li_grad_gate = 1.0f;

				#if (LIGHT_LI_CLAMP > 0)
					li_grad_gate = (Lout.li_clamped > 0.5f) ? 0.0f : 1.0f;
				#endif

				// ---- intensity learnable (per-scene) gradient ----
				{
					// Li = I * inv  => dLi/dI = inv
					float dL_dI = (dL_dLi * li_grad_gate) * Lout.inv;

					// I = softplus_beta2(clamp(I_raw, -15, 15))
					// Lout.dI_raw already stores dI/dI_raw including the clamp gate
					float dL_dIraw = dL_dI * Lout.dI_raw;

					// per-scene scalar gradient
				#if (LIGHT_INTENSITY_MODE == 1)
					atomicAdd(&dL_dintensity_raw[0], dL_dIraw);
				#endif
				}

			#if FALLOFF_Z_GRAD_ENABLE
				float contrib = (dL_dLi * li_grad_gate) * Lout.dintensity_ddepth;
				contrib *= FALLOFF_Z_GRAD_SCALE;
				contrib = fminf(fmaxf(contrib, -FALLOFF_Z_GRAD_CLAMP), FALLOFF_Z_GRAD_CLAMP);
				dL_dz += contrib;
			#endif
			}

			// ---------- Ambient gradient (learned only) ----------
			#if LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
			{
				float d_diffuse_da = 1.0f;

				float t = sigmoidf_stable(ambients[0]);
				float da_draw = t * (1.0f - t);

				dAmb += dL_dindirect_approx  * d_diffuse_da * da_draw;
			}
			#endif

			// ---------- Roughness gradient (bridge: D-term + G-term) ----------
			#if LIGHT_USE_PHONG
			{
				if ((dL_dspec_rgb.x != 0.0f || dL_dspec_rgb.y != 0.0f || dL_dspec_rgb.z != 0.0f) &&
					Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
				{
					const float nh = fmaxf(Lout.ndoth, 1e-6f);
					const float a2 = fmaxf(Lout.alpha2, 1e-8f);

					// D = a2 / (pi * t^2 + eps), t = nh^2 (a2 - 1) + 1
					const float t = nh * nh * (a2 - 1.0f) + 1.0f;
					const float denom = LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS;

					// derivative of D wrt a2
					const float dD_da2 =
						(denom - a2 * (2.0f * LIGHT_PI * t * nh * nh)) / (denom * denom);

					const float nv = fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS);
					const float nl = fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS);

					// spec scalar proxy = avg(F_rgb) * D * G / (4 nv nl)
					const float3 dspec_da2_rgb = make_float3(
						(Lout.fresnel_rgb.x * Lout.G / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS)) * dD_da2 * Lout.lambert * Lout.spot * Lout.Li,
						(Lout.fresnel_rgb.y * Lout.G / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS)) * dD_da2 * Lout.lambert * Lout.spot * Lout.Li,
						(Lout.fresnel_rgb.z * Lout.G / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS)) * dD_da2 * Lout.lambert * Lout.spot * Lout.Li
					);

					// alpha2 = roughness^4  => d(alpha2)/d(roughness) = 4 r^3
					const float r     = fmaxf(Lout.roughness, 1e-6f);
					const float alpha = r * r;
					const float3 dspec_dalpha_from_D_rgb = make_float3(
						dspec_da2_rgb.x * 2.0f * alpha,
						dspec_da2_rgb.y * 2.0f * alpha,
						dspec_da2_rgb.z * 2.0f * alpha
					);

					// Smith G term derivative wrt roughness
					auto dG1_dr = [](float nx, float r) -> float
					{
						nx = fmaxf(nx, 1e-6f);
						r  = fmaxf(r,  1e-6f);

						// k = (r + 1)^2 / 8
						const float k = ((r + 1.0f) * (r + 1.0f)) * 0.125f;
						const float dk_dr = 0.25f * (r + 1.0f);

						// G1 = nx / (nx * (1-k) + k)
						const float denom = nx * (1.0f - k) + k;
						const float ddenom_dr = (1.0f - nx) * dk_dr;

						return -nx * ddenom_dr / (denom * denom);
					};

					const float dGv_dr = dG1_dr(nv, r);
					const float dGl_dr = dG1_dr(nl, r);
					const float dG_dr  = dGv_dr * Lout.Gl + Lout.Gv * dGl_dr;

					const float3 dspec_dG_rgb = make_float3(
						(Lout.fresnel_rgb.x * Lout.D / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS)) * Lout.lambert * Lout.spot * Lout.Li,
						(Lout.fresnel_rgb.y * Lout.D / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS)) * Lout.lambert * Lout.spot * Lout.Li,
						(Lout.fresnel_rgb.z * Lout.D / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS)) * Lout.lambert * Lout.spot * Lout.Li
					);

					// G path: d(spec)/d(alpha) = d(spec)/d(roughness) * d(roughness)/d(alpha) = dG_dr / (2r)
					const float dr_dalpha = 0.5f / r;
					const float3 dspec_dalpha_from_G_rgb = make_float3(
						dspec_dG_rgb.x * dG_dr * dr_dalpha,
						dspec_dG_rgb.y * dG_dr * dr_dalpha,
						dspec_dG_rgb.z * dG_dr * dr_dalpha
					);

					// dL/d(alpha);  Lout.drough_raw = d(alpha)/d(raw)
					const float dL_dalpha =
						dL_dspec_rgb.x * (dspec_dalpha_from_D_rgb.x + dspec_dalpha_from_G_rgb.x) +
						dL_dspec_rgb.y * (dspec_dalpha_from_D_rgb.y + dspec_dalpha_from_G_rgb.y) +
						dL_dspec_rgb.z * (dspec_dalpha_from_D_rgb.z + dspec_dalpha_from_G_rgb.z);

					const float dL_draw = dL_dalpha * Lout.drough_raw;

					atomicAdd(&dL_droughness[global_id], dL_draw);
				}
			}
			#endif

			// ---------- Metallic gradient (bridge: diffuse + F0/spec proxy) ----------
			#if LIGHT_USE_PHONG
			{
				float dL_dm = 0.0f;

				#if LIGHT_USE_LAMBERT
				{
					const float x = 1.0f - fmaxf(Lout.vdoth, 0.0f);
					const float x2 = x * x;
					const float x5 = x2 * x2 * x;
					const float dF_dF0 = 1.0f - x5;

					const float3 dF0_dm = make_float3(
						base_rgb.x - LIGHT_GGX_F0_DIELECTRIC,
						base_rgb.y - LIGHT_GGX_F0_DIELECTRIC,
						base_rgb.z - LIGHT_GGX_F0_DIELECTRIC
					);

					const float3 dF_dm = make_float3(
						dF_dF0 * dF0_dm.x,
						dF_dF0 * dF0_dm.y,
						dF_dF0 * dF0_dm.z
					);

					const float3 dkd_dm = make_float3(
						-(1.0f - Lout.fresnel_rgb.x) - (1.0f - Lout.metallic) * dF_dm.x,
						-(1.0f - Lout.fresnel_rgb.y) - (1.0f - Lout.metallic) * dF_dm.y,
						-(1.0f - Lout.fresnel_rgb.z) - (1.0f - Lout.metallic) * dF_dm.z
					);

					const float diffuse_light = Lout.indirect_diffuse + Lout.direct_diffuse_raw;

					const float3 dDiffuse_dm_rgb = make_float3(
						diffuse_light * dkd_dm.x,
						diffuse_light * dkd_dm.y,
						diffuse_light * dkd_dm.z
					);

					dL_dm += dL_ddiffuse_rgb.x * dDiffuse_dm_rgb.x;
					dL_dm += dL_ddiffuse_rgb.y * dDiffuse_dm_rgb.y;
					dL_dm += dL_ddiffuse_rgb.z * dDiffuse_dm_rgb.z;
				}
				#endif

				// spec path through F0
				if ((dL_dspec_rgb.x != 0.0f || dL_dspec_rgb.y != 0.0f || dL_dspec_rgb.z != 0.0f) &&
					Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
				{
					// F0_rgb = 0.04*(1-m) + base_rgb*m  => dF0/dm = base_rgb - 0.04
					const float3 base_f0 = make_float3(
						saturate01(base_rgb.x),
						saturate01(base_rgb.y),
						saturate01(base_rgb.z)
					);

					const float3 dF0_dm = make_float3(
						base_f0.x - LIGHT_GGX_F0_DIELECTRIC,
						base_f0.y - LIGHT_GGX_F0_DIELECTRIC,
						base_f0.z - LIGHT_GGX_F0_DIELECTRIC
					);

					// F = F0 + (1-F0)(1-vh)^5 = F0*(1-k) + k, so dF/dF0 = 1 - (1-vh)^5
					const float x = 1.0f - fmaxf(Lout.vdoth, 0.0f);
					const float x2 = x * x;
					const float x5 = x2 * x2 * x;
					const float dF_dF0 = 1.0f - x5;

					const float nv = fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS);
					const float nl = fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS);
					const float common = (Lout.D * Lout.G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);

					const float dspec_dm_r = dF0_dm.x * dF_dF0 * common * Lout.lambert * Lout.spot * Lout.Li;
					const float dspec_dm_g = dF0_dm.y * dF_dF0 * common * Lout.lambert * Lout.spot * Lout.Li;
					const float dspec_dm_b = dF0_dm.z * dF_dF0 * common * Lout.lambert * Lout.spot * Lout.Li;

					dL_dm += dL_dspec_rgb.x * dspec_dm_r;
					dL_dm += dL_dspec_rgb.y * dspec_dm_g;
					dL_dm += dL_dspec_rgb.z * dspec_dm_b;
				}

				const float dL_dmraw = dL_dm * Lout.dmetal_raw;
				atomicAdd(&dL_dmetallic[global_id], dL_dmraw);
			}
			#endif

			// ---------- Normal gradient approximation ----------
			if (dL_ddiffuse_rgb.x != 0.0f || dL_ddiffuse_rgb.y != 0.0f || dL_ddiffuse_rgb.z != 0.0f ||
				dL_dspec_rgb.x != 0.0f || dL_dspec_rgb.y != 0.0f || dL_dspec_rgb.z != 0.0f)
			{
				// Unit normal used in forward
				float3 view_ray = normalize_or_default(point_cam, make_float3(0.f, 0.f, 1.f));

				const float3 light_pos = make_float3(0.0f, 0.0f, 0.0f);

				float3 Lvec = make_float3(light_pos.x - point_cam.x,
										light_pos.y - point_cam.y,
										light_pos.z - point_cam.z);
				Lvec = normalize_or_default(Lvec, make_float3(0.f, 0.f, -1.f));

				float3 Vvec = make_float3(-view_ray.x, -view_ray.y, -view_ray.z);
				Vvec = normalize_or_default(Vvec, make_float3(0.f, 0.f, -1.f));

				// Half vector (same as forward)
				float3 Hh = normalize_or_default(
					make_float3(Lvec.x + Vvec.x, Lvec.y + Vvec.y, Lvec.z + Vvec.z),
					make_float3(0.f, 0.f, -1.f)
				);

				float3 g_unit = make_float3(0.f, 0.f, 0.f);

				// lambert ndotl contribution
				float dL_dndotl = 0.0f;
				#if LIGHT_USE_LAMBERT
				{
					float dL_dlambert = 0.0f;
					if (Lout.lambert > 1e-6f)
					{
						dL_dlambert =
							dL_ddiffuse_rgb.x * (Lout.direct_diffuse_rgb.x / Lout.lambert) +
							dL_ddiffuse_rgb.y * (Lout.direct_diffuse_rgb.y / Lout.lambert) +
							dL_ddiffuse_rgb.z * (Lout.direct_diffuse_rgb.z / Lout.lambert);
					}

					#if LIGHT_LAMBERT_ABS
						if (Lout.ndotl > 0.0f) dL_dndotl += dL_dlambert;
						else if (Lout.ndotl < 0.0f) dL_dndotl -= dL_dlambert;
					#else
						if (Lout.ndotl > 0.0f) dL_dndotl += dL_dlambert;
					#endif
				}
				#endif

				#if LIGHT_USE_PHONG && (LIGHT_SPEC_GATING == 2)
				{
					if (dL_dspec_rgb.x != 0.0f || dL_dspec_rgb.y != 0.0f || dL_dspec_rgb.z != 0.0f)
					{
						float dL_dlambert_from_spec =
							dL_dspec_rgb.x * Lout.spec_dir_raw_rgb.x +
							dL_dspec_rgb.y * Lout.spec_dir_raw_rgb.y +
							dL_dspec_rgb.z * Lout.spec_dir_raw_rgb.z;

						#if LIGHT_LAMBERT_ABS
							if (Lout.ndotl > 0.0f) dL_dndotl += dL_dlambert_from_spec;
							else if (Lout.ndotl < 0.0f) dL_dndotl -= dL_dlambert_from_spec;
						#else
							if (Lout.ndotl > 0.0f) dL_dndotl += dL_dlambert_from_spec;
						#endif
					}
				}
				#endif

				if (dL_dndotl != 0.0f) {
					g_unit.x += dL_dndotl * Lvec.x;
					g_unit.y += dL_dndotl * Lvec.y;
					g_unit.z += dL_dndotl * Lvec.z;
				}

				// Specular normal gradient: tighter GGX derivative
				#if LIGHT_USE_PHONG
				{
					if ((dL_dspec_rgb.x != 0.0f || dL_dspec_rgb.y != 0.0f || dL_dspec_rgb.z != 0.0f) &&
						Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
					{
						const float nv = fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS);
						const float nl = fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS);
						const float nh = fmaxf(Lout.ndoth, 1e-6f);
						const float a2 = fmaxf(Lout.alpha2, 1e-8f);

						const float spotLi = Lout.spot * Lout.Li;

						// Cook-Torrance denominator
						const float denom_raw = 4.0f * nv * nl;
						const float denom = fmaxf(denom_raw, LIGHT_GGX_DENOM_EPS);
						const float inv_denom = 1.0f / denom;

						// ---------------- D term ----------------
						// D = a2 / (pi * t^2 + eps),  t = nh^2 (a2 - 1) + 1
						const float t = nh * nh * (a2 - 1.0f) + 1.0f;
						const float Dden = LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS;

						float dD_dnh_val;
						ggx_D_and_dDdnh(Lout.ndoth, Lout.alpha2, &dD_dnh_val);

						// ---------------- G term ----------------
						// G = Gv * Gl
						// G1(nx) = nx / (nx(1-k) + k + eps)
						const float r = fmaxf(Lout.roughness, 1e-6f);
						const float k = ((r + 1.0f) * (r + 1.0f)) * 0.125f;

						const float denom_v = nv * (1.0f - k) + k + LIGHT_GGX_DENOM_EPS;
						const float denom_l = nl * (1.0f - k) + k + LIGHT_GGX_DENOM_EPS;

						const float dGv_dnv = (k + LIGHT_GGX_DENOM_EPS) / fmaxf(denom_v * denom_v, 1e-12f);
						const float dGl_dnl = (k + LIGHT_GGX_DENOM_EPS) / fmaxf(denom_l * denom_l, 1e-12f);

						const float dG_dnv = Lout.Gl * dGv_dnv;
						const float dG_dnl = Lout.Gv * dGl_dnl;

						// ---------------- common = D * G / denom ----------------
						float dcommon_dnh = dD_dnh_val * Lout.G * inv_denom;
						float dcommon_dnv = 0.0f;
						float dcommon_dnl = 0.0f;

						if (denom_raw > LIGHT_GGX_DENOM_EPS)
						{
							const float inv_denom2 = inv_denom * inv_denom;

							dcommon_dnv =
								Lout.D * (dG_dnv * inv_denom - Lout.G * (4.0f * nl) * inv_denom2);

							dcommon_dnl =
								Lout.D * (dG_dnl * inv_denom - Lout.G * (4.0f * nv) * inv_denom2);
						}
						else
						{
							// rare clamp case: keep only G derivative contribution
							dcommon_dnv = Lout.D * dG_dnv * inv_denom;
							dcommon_dnl = Lout.D * dG_dnl * inv_denom;
						}

						const float3 dspec_dndoth_rgb = make_float3(
							Lout.fresnel_rgb.x * dcommon_dnh * spotLi,
							Lout.fresnel_rgb.y * dcommon_dnh * spotLi,
							Lout.fresnel_rgb.z * dcommon_dnh * spotLi
						);

						const float3 dspec_dndotv_rgb = make_float3(
							Lout.fresnel_rgb.x * dcommon_dnv * spotLi,
							Lout.fresnel_rgb.y * dcommon_dnv * spotLi,
							Lout.fresnel_rgb.z * dcommon_dnv * spotLi
						);

						const float3 dspec_dndotl_rgb = make_float3(
							Lout.fresnel_rgb.x * dcommon_dnl * spotLi,
							Lout.fresnel_rgb.y * dcommon_dnl * spotLi,
							Lout.fresnel_rgb.z * dcommon_dnl * spotLi
						);

						float3 dspec_dndoth_rgb_g = dspec_dndoth_rgb;
						float3 dspec_dndotv_rgb_g = dspec_dndotv_rgb;
						float3 dspec_dndotl_rgb_g = dspec_dndotl_rgb;

						#if (LIGHT_SPEC_GATING == 1)
							if (Lout.ndotl <= 0.0f)
							{
								dspec_dndoth_rgb_g = make_float3(0.0f, 0.0f, 0.0f);
								dspec_dndotv_rgb_g = make_float3(0.0f, 0.0f, 0.0f);
								dspec_dndotl_rgb_g = make_float3(0.0f, 0.0f, 0.0f);
							}
						#elif (LIGHT_SPEC_GATING == 2)
							// The derivative of the lambert gate itself is already handled
							// in the earlier dL_dlambert_from_spec block. So here we only
							// scale the raw GGX derivatives by lambert.
							dspec_dndoth_rgb_g = make_float3(
								dspec_dndoth_rgb_g.x * Lout.lambert,
								dspec_dndoth_rgb_g.y * Lout.lambert,
								dspec_dndoth_rgb_g.z * Lout.lambert
							);
							dspec_dndotv_rgb_g = make_float3(
								dspec_dndotv_rgb_g.x * Lout.lambert,
								dspec_dndotv_rgb_g.y * Lout.lambert,
								dspec_dndotv_rgb_g.z * Lout.lambert
							);
							dspec_dndotl_rgb_g = make_float3(
								dspec_dndotl_rgb_g.x * Lout.lambert,
								dspec_dndotl_rgb_g.y * Lout.lambert,
								dspec_dndotl_rgb_g.z * Lout.lambert
							);
						#endif

						const float dL_dndoth =
							dL_dspec_rgb.x * dspec_dndoth_rgb_g.x +
							dL_dspec_rgb.y * dspec_dndoth_rgb_g.y +
							dL_dspec_rgb.z * dspec_dndoth_rgb_g.z;

						const float dL_dndotv_from_spec =
							dL_dspec_rgb.x * dspec_dndotv_rgb_g.x +
							dL_dspec_rgb.y * dspec_dndotv_rgb_g.y +
							dL_dspec_rgb.z * dspec_dndotv_rgb_g.z;

						const float dL_dndotl_from_spec =
							dL_dspec_rgb.x * dspec_dndotl_rgb_g.x +
							dL_dspec_rgb.y * dspec_dndotl_rgb_g.y +
							dL_dspec_rgb.z * dspec_dndotl_rgb_g.z;

						if (dL_dndoth != 0.0f)
						{
							g_unit.x += dL_dndoth * Hh.x;
							g_unit.y += dL_dndoth * Hh.y;
							g_unit.z += dL_dndoth * Hh.z;
						}

						if (dL_dndotv_from_spec != 0.0f)
						{
							g_unit.x += dL_dndotv_from_spec * Vvec.x;
							g_unit.y += dL_dndotv_from_spec * Vvec.y;
							g_unit.z += dL_dndotv_from_spec * Vvec.z;
						}

						if (dL_dndotl_from_spec != 0.0f)
						{
							g_unit.x += dL_dndotl_from_spec * Lvec.x;
							g_unit.y += dL_dndotl_from_spec * Lvec.y;
							g_unit.z += dL_dndotl_from_spec * Lvec.z;
						}
					}
				}
				#endif

				// back through normalize(n_raw) -> n
				#if LIGHT_USE_SHADING_NORMAL
					// Ns comes from basis, so do not push this BRDF normal gradient into n_raw.
					float3 g_raw = make_float3(0.0f, 0.0f, 0.0f);
				#else
					float3 g_raw = apply_norm_jacobian(n_raw, g_unit);
				#endif

				const float gmax = 5.0f;
				g_raw.x = fminf(fmaxf(g_raw.x, -gmax), gmax);
				g_raw.y = fminf(fmaxf(g_raw.y, -gmax), gmax);
				g_raw.z = fminf(fmaxf(g_raw.z, -gmax), gmax);

				atomicAdd(&dL_dnormal3D[global_id * 3 + 0], g_raw.x);
				atomicAdd(&dL_dnormal3D[global_id * 3 + 1], g_raw.y);
				atomicAdd(&dL_dnormal3D[global_id * 3 + 2], g_raw.z);

				// ---------- point_cam lighting gradient approximation ----------
				{
					float3 Ns = normalize_or_default(cross(bu_cam, bv_cam), normalize_or_default(n_raw, make_float3(0.f, 0.f, 1.f)));
					float3 n_for_geom = Ns;

					float3 g_point_light = pointcam_lighting_grad_approx(
						Lout,
						point_cam,
						Ns,
						dL_ddiffuse_rgb,
						dL_dspec_rgb
					);

					extra_dL_dsx += g_point_light.x * bu_cam.x +
									g_point_light.y * bu_cam.y +
									g_point_light.z * bu_cam.z;

					extra_dL_dsy += g_point_light.x * bv_cam.x +
									g_point_light.y * bv_cam.y +
									g_point_light.z * bv_cam.z;
				}
			}

			#else
				// No lighting, original 2DGS
				const float w = alpha * T;
				const float dchannel_dcolor = w;

				for (int ch = 0; ch < C; ch++) {
					const float c = collected_colors[ch * BLOCK_SIZE + j];
					const float dL_dchannel = dL_dpixel[ch];

					accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = c;

					float eff = c; // diffuse_mul == 1
					accum_eff[ch] = last_alpha * last_eff[ch] + (1.f - last_alpha) * accum_eff[ch];
					last_eff[ch] = eff;

					dL_dalpha += (eff - accum_eff[ch]) * dL_dchannel;

					atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				}
			#endif

		// ========== END LAMBERT + PHONG SHADING (BACKWARD) ========================

        float dL_dweight = 0;

			#if RENDER_AXUTILITY
			if (depth_valid)
			{
				const float m_d = far_n / (far_n - near_n) * (1.0f - near_n / c_d);
				const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);

				if (contributor == median_contributor - 1)
				{
					dL_dz += dL_dmedian_depth;
				}

			#if DETACH_WEIGHT
				dL_dweight += 0.0f;
			#else
				dL_dweight += (final_D2 + m_d * m_d * final_A - 2.0f * m_d * final_D) * dL_dreg;
			#endif

				dL_dalpha_aux += dL_dweight - last_dL_dT;
				last_dL_dT = dL_dweight * alpha + (1.0f - alpha) * last_dL_dT;

				const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
				dL_dz += dL_dmd * dmd_dd;

				accum_depth_rec = last_alpha * last_depth + (1.0f - last_alpha) * accum_depth_rec;
				last_depth = c_d;
				dL_dalpha_aux += (c_d - accum_depth_rec) * dL_ddepth;

				accum_alpha_rec = last_alpha * 1.0f + (1.0f - last_alpha) * accum_alpha_rec;
				dL_dalpha_aux += (1.0f - accum_alpha_rec) * dL_daccum;

				for (int ch = 0; ch < 3; ch++)
				{
					accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.0f - last_alpha) * accum_normal_rec[ch];
					last_normal[ch] = normal[ch];
					dL_dalpha_aux += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
					atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
				}

				dL_dz += alpha * T * dL_ddepth;
			}
			#endif

			#if LIGHT_SURFACE_SHADING_MODE

				dL_dalpha = dL_dalpha_surface;

				#if RENDER_AXUTILITY
				dL_dalpha += T * dL_dalpha_aux;
				#endif

			#else

				dL_dalpha *= T;

				#if RENDER_AXUTILITY
				dL_dalpha += T * dL_dalpha_aux;
				#endif

			#endif
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			#if !LIGHT_SURFACE_SHADING_MODE
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			#endif


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;

			if (use_3d_footprint) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};

				dL_ds.x += extra_dL_dsx;
				dL_ds.y += extra_dL_dsy;

				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
			} else {
				// Update gradients w.r.t. center of Gaussian 2D mean position

				// Correct indices for x and y gradients
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;

				atomicAdd(&dL_dmean2D[global_id].x, (dL_dG * dG_ddelx));	// Update X
				atomicAdd(&dL_dmean2D[global_id].y, (dL_dG * dG_ddely));   // Update Y

				// Propagate the gradients of depth
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  s.x * dL_dz);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  s.y * dL_dz);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
		}

	#if LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
	{
		// Block-level reduction before atomicAdd to reduce contention
		// on the single per-scene ambient scalar
		__shared__ float amb_reduce[BLOCK_SIZE];
		amb_reduce[block.thread_rank()] = dAmb;
		block.sync();

		// Simple tree reduction
		for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
		{
			if (block.thread_rank() < stride)
				amb_reduce[block.thread_rank()] += amb_reduce[block.thread_rank() + stride];
			block.sync();
		}

		if (block.thread_rank() == 0)
			atomicAdd(&dL_dambients[0], amb_reduce[0]);
	}
	#endif
}

__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	const glm::vec3* dL_dbasis_u_cam,
	const glm::vec3* dL_dbasis_v_cam)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	// Surface-BRDF normal gradient path:
	// basis_u_cam = view_rot * L[0]
	// basis_v_cam = view_rot * L[1]
	//
	// Convert camera-space basis gradients back to world-space
	// and add them to dL_dRS columns 0 and 1.

	glm::mat3 view_rot = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]
	);

	glm::vec3 gbu_cam = dL_dbasis_u_cam[idx];
	glm::vec3 gbv_cam = dL_dbasis_v_cam[idx];

	glm::vec3 gbu_world = glm::transpose(view_rot) * gbu_cam;
	glm::vec3 gbv_world = glm::transpose(view_rot) * gbv_cam;

	dL_dRS[0] += gbu_world;
	dL_dRS[1] += gbv_world;

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	glm::vec3* dL_dbasis_u_cam,
	glm::vec3* dL_dbasis_v_cam)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots,
		dL_dbasis_u_cam,
    	dL_dbasis_v_cam
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	glm::vec3* dL_dbasis_u_cam,
	glm::vec3* dL_dbasis_v_cam)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots,
		dL_dbasis_u_cam,
		dL_dbasis_v_cam
	);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float2* means2D,
	const float4* normal_opacity,
	const float* colors,
	const float* ambients,
	const float* intensity,
	const float* roughness,
	const float* metallic,
	const float* transMats,
	const float* depths,
	const float3* means3D_cam,
	const float3* basis_u_cam,
    const float3* basis_v_cam,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dtransMat,
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dambient,
	float* dL_dintensity,
	float* dL_droughness,
	float* dL_dmetallic,
	float3* dL_dbasis_u_cam,
	float3* dL_dbasis_v_cam)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		colors,
		ambients,
		intensity,
		roughness,
		metallic,
		depths,
		means3D_cam,
		basis_u_cam,
		basis_v_cam,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors,
		dL_dambient,
		dL_dintensity,
		dL_droughness,
		dL_dmetallic,
		dL_dbasis_u_cam,
		dL_dbasis_v_cam
		);
}