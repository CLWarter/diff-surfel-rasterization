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
	float* __restrict__ dL_dmetallic)
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

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// per-thread accumulators
	float dAmb = 0.0f;
	float dSh = 0.0f;

	// shared buffers for block reduction
	__shared__ float amb_reduce[BLOCK_SIZE];

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
			float rho = min(rho3d, rho2d);

			float3 point_cam = make_float3(
				center_cam.x + s.x * bu_cam.x + s.y * bv_cam.x,
				center_cam.y + s.x * bu_cam.y + s.y * bv_cam.y,
				center_cam.z + s.x * bu_cam.z + s.y * bv_cam.z
			);

			// compute depth
			float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z; // Tw * [u,v,1]
			// if a point is too small, its depth is not reliable?
			// c_d = (rho3d <= rho2d) ? c_d : Tw.z; 
			if (c_d < near_n) continue;
			
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
			const int global_id = collected_id[j];

			// depth accumulator
            float dL_dz      = 0.0f;

			float extra_dL_dsx = 0.0f;
			float extra_dL_dsy = 0.0f;

			LightingOut Lout = {};
			const float* rough_ptr = nullptr;
			const float* metal_ptr = nullptr;

			#if LIGHT_ENABLE_BWD && (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)

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
					&point_cam
				);

				const float w = alpha * T;

				// Shaded color is: shaded = diffuse_mul * base_color + spec_add (RGB only)
				// Gradient into base color multiplied by diffuse_mul
				const float dchannel_dcolor = w * (Lout.indirect_diffuse + Lout.direct_diffuse);

				// Accumulators for lighting parameter gradients
				float dL_ddiffuse = 0.0f; // accum over channels
				float dL_dspec    = 0.0f; // accum over RGB only

				#if LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
					float dL_ddiffuse_shading = 0.0f; // accum for ambient gradient
				#endif

				for (int ch = 0; ch < C; ch++)
				{
					const float c = collected_colors[ch * BLOCK_SIZE + j];
					const float dL_dchannel = dL_dpixel[ch];

					// ambient gradient
					#if LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
						dL_ddiffuse_shading += dL_dchannel * (w * c);
					#endif

					// reccurence
					accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = c;

					float eff = (Lout.indirect_diffuse + Lout.direct_diffuse) * c;
					#if LIGHT_USE_PHONG
						if (ch == 0) eff += Lout.spec_add_rgb.x;
						else if (ch == 1) eff += Lout.spec_add_rgb.y;
						else if (ch == 2) eff += Lout.spec_add_rgb.z;
					#endif

					accum_eff[ch] = last_alpha * last_eff[ch] + (1.f - last_alpha) * accum_eff[ch];
					last_eff[ch] = eff;

					// alpha influence contribution of gaussians to accumulated behind
					dL_dalpha += (eff - accum_eff[ch]) * dL_dchannel;

					// gradients needed for spec path
					dL_ddiffuse += dL_dchannel * (w * c);
					#if LIGHT_USE_PHONG
						if (ch < 3) dL_dspec += dL_dchannel * w * (1.0f / 3.0f); // see spec_add = avg(spec_add_rgb)
					#endif

					// base color gradient
					atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);

					#if LIGHT_USE_PHONG
						if (ch < 3 && Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
						{
							// baseColor affects F0 only through metallic workflow
							const float x = 1.0f - fmaxf(Lout.vdoth, 0.0f);
							const float x2 = x * x;
							const float x5 = x2 * x2 * x;
							const float dF_dF0 = 1.0f - x5;

							const float nv = fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS);
							const float nl = fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS);
							const float common = (Lout.D * Lout.G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);

							// F0_channel = 0.04*(1-m) + baseColor_channel * m
							const float dF0_dbase = Lout.metallic;

							const float dspec_dbase =
								dF0_dbase * dF_dF0 * common * Lout.spot * Lout.Li;

							atomicAdd(&(dL_dcolors[global_id * C + ch]), dL_dpixel[ch] * w * dspec_dbase);
						}
					#endif
				}

			// -------- intensity (falloff) to depth gradient --------
			{
				float dDiff_dLi = 0.0f;
				#if LIGHT_USE_LAMBERT
					// only directional diffuse depends on Li
					dDiff_dLi = Lout.diffuse_brdf * Lout.lambert * Lout.spot;

					// New BRDF path:
					// diffuse attenuation is no longer driven by legacy kspec.
				#endif

				float dSpec_dLi = 0.0f;
				#if LIGHT_USE_PHONG
					// scalar proxy of RGB GGX spec wrt Li
					float base = Lout.spec_add / fmaxf(Lout.Li, 1e-6f);

					#if (LIGHT_SPEC_GATING == 1)
						if (Lout.ndotl <= 0.0f) base = 0.0f;
					#endif

					dSpec_dLi = base;
				#endif

				float dL_dLi = dL_ddiffuse * dDiff_dLi + dL_dspec * dSpec_dLi;

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

				const float amax = 0.25f;
				float t = sigmoidf_stable(ambients[0]);
				float da_draw = amax * t * (1.0f - t);

				dAmb += dL_ddiffuse_shading * d_diffuse_da * da_draw;
			}
			#endif

			// ---------- Roughness gradient (bridge: D-term + G-term) ----------
			#if LIGHT_USE_PHONG
			{
				if (dL_dspec != 0.0f && Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
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
					const float dspec_da2 =
						(Lout.fresnel * Lout.G / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS))
						* dD_da2 * Lout.spot * Lout.Li;

					// alpha2 = roughness^4  => d(alpha2)/d(roughness) = 4 r^3
					const float r  = fmaxf(Lout.roughness, 1e-6f);
					const float da2_dr = 4.0f * r * r * r;
					const float dspec_dr_from_D = dspec_da2 * da2_dr;

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

					const float dspec_dG =
						(Lout.fresnel * Lout.D / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS))
						* Lout.spot * Lout.Li;

					const float dspec_dr_from_G = dspec_dG * dG_dr;
					const float dspec_dr = dspec_dr_from_D + dspec_dr_from_G;

					const float dL_dr = dL_dspec * dspec_dr;
					const float dL_draw = dL_dr * Lout.drough_raw;

					atomicAdd(&dL_droughness[global_id], dL_draw);
				}
			}
			#endif

			// ---------- Metallic gradient (bridge: diffuse + F0/spec proxy) ----------
			#if LIGHT_USE_PHONG
			{
				float dL_dm = 0.0f;

				// diffuse path: direct_diffuse = raw * kd, kd ~= avg((1-F)*(1-m))
				#if LIGHT_USE_LAMBERT
				{
					float kd = 1.0f;
					if (Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
					{
						kd = float3_avg(make_float3(
							(1.0f - Lout.fresnel_rgb.x) * (1.0f - Lout.metallic),
							(1.0f - Lout.fresnel_rgb.y) * (1.0f - Lout.metallic),
							(1.0f - Lout.fresnel_rgb.z) * (1.0f - Lout.metallic)
						));
					}

					const float dkd_dm = -float3_avg(make_float3(
						1.0f - Lout.fresnel_rgb.x,
						1.0f - Lout.fresnel_rgb.y,
						1.0f - Lout.fresnel_rgb.z
					));

					const float dDiffuse_dm = Lout.direct_diffuse_raw * dkd_dm;
					dL_dm += dL_ddiffuse * dDiffuse_dm;
				}
				#endif

				// spec path through F0
				if (dL_dspec != 0.0f && Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
				{
					// F0_rgb = 0.04*(1-m) + base_rgb*m  => dF0/dm = base_rgb - 0.04
					const float3 dF0_dm = make_float3(
						base_rgb.x - LIGHT_GGX_F0_DIELECTRIC,
						base_rgb.y - LIGHT_GGX_F0_DIELECTRIC,
						base_rgb.z - LIGHT_GGX_F0_DIELECTRIC
					);

					// F = F0 + (1-F0)(1-vh)^5 = F0*(1-k) + k, so dF/dF0 = 1 - (1-vh)^5
					const float x = 1.0f - fmaxf(Lout.vdoth, 0.0f);
					const float x2 = x * x;
					const float x5 = x2 * x2 * x;
					const float dF_dF0 = 1.0f - x5;

					const float nv = fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS);
					const float nl = fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS);
					const float common = (Lout.D * Lout.G) / fmaxf(4.0f * nv * nl, LIGHT_GGX_DENOM_EPS);

					const float dspec_dm =
						float3_avg(make_float3(
							dF0_dm.x * dF_dF0,
							dF0_dm.y * dF_dF0,
							dF0_dm.z * dF_dF0
						)) * common * Lout.spot * Lout.Li;

					dL_dm += dL_dspec * dspec_dm;
				}

				const float dL_dmraw = dL_dm * Lout.dmetal_raw;
				atomicAdd(&dL_dmetallic[global_id], dL_dmraw);
			}
			#endif

			// Step 2 metallic-roughness bridge:
			// legacy kspec no longer drives the main BRDF, so disable its gradient here.

			// Step 1 GGX bridge:
			// no shininess learning anymore; roughness is fixed for now.

			// ---------- Normal gradient approximation ----------
			if (dL_ddiffuse != 0.0f || dL_dspec != 0.0f)
			{
				// Unit normal used in forward
				float3 view_ray = normalize_or_default(point_cam, make_float3(0.f, 0.f, 1.f));

				float3 n = normalize_or_default(n_raw, make_float3(0.f, 0.f, 1.f));

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
					float dL_dlambert = dL_ddiffuse * Lout.diffuse_brdf * Lout.spot * Lout.intensity;
					// New BRDF path:
					// no legacy kspec attenuation on diffuse lambert term.

					#if LIGHT_LAMBERT_ABS
						if (Lout.ndotl > 0.0f) dL_dndotl += dL_dlambert;
						else if (Lout.ndotl < 0.0f) dL_dndotl -= dL_dlambert;
					#else
						if (Lout.ndotl > 0.0f) dL_dndotl += dL_dlambert; // max(ndotl,0)
					#endif
				}
				#endif

				#if LIGHT_USE_PHONG && (LIGHT_SPEC_GATING == 2)
				{
					if (dL_dspec != 0.0f)
					{
						float dspec_dlambert = Lout.spec_dir_raw;

						float dL_dlambert_from_spec = dL_dspec * dspec_dlambert;

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

				// Specular ndoth contribution
				#if LIGHT_USE_PHONG
				{
					const float NDOTH_MAX = 0.95f;
					if (dL_dspec != 0.0f && Lout.ndotl > 0.0f && Lout.ndotv > 0.0f)
					{
						// Step 1 bridge approximation:
						// treat GGX D/G/F as locally frozen with respect to n except through NdotL / NdotH.
						// This is not the final full derivative, but it is a practical bridge.
						float dspec_dndoth = 0.0f;
						float dspec_dndotl = 0.0f;

						// D term sensitivity wrt NdotH
						{
						 const float nh = fmaxf(Lout.ndoth, 1e-6f);
						 const float a2 = Lout.alpha2;
						 const float t = nh * nh * (a2 - 1.0f) + 1.0f;
						 const float dD_dnh =
							 (-4.0f * LIGHT_PI * a2 * nh * (a2 - 1.0f) * t) /
							 fmaxf((LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS) * (LIGHT_PI * t * t + LIGHT_GGX_DENOM_EPS), 1e-8f);

						 float pref = (Lout.G * Lout.fresnel) /
							 fmaxf(4.0f * fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS) * fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS), LIGHT_GGX_DENOM_EPS);

						 dspec_dndoth = dD_dnh * pref * Lout.spot * Lout.intensity;
						}

						// denominator sensitivity wrt NdotL
						{
						 float denom = fmaxf(4.0f * fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS) * fmaxf(Lout.ndotl, LIGHT_GGX_NL_EPS), LIGHT_GGX_DENOM_EPS);
						 float numer = Lout.D * Lout.G * Lout.fresnel;
						 float dspecbrdf_dnl = -numer * (4.0f * fmaxf(Lout.ndotv, LIGHT_GGX_NV_EPS)) / (denom * denom);
						 dspec_dndotl = dspecbrdf_dnl * Lout.spot * Lout.intensity;
						}

						#if (LIGHT_SPEC_GATING == 1)
							if (Lout.ndotl <= 0.0f) {
								dspec_dndoth = 0.0f;
								dspec_dndotl = 0.0f;
							}
						#elif (LIGHT_SPEC_GATING == 2)
							dspec_dndoth *= Lout.lambert;
							// plus lambert path contribution
							dspec_dndotl += Lout.spec_dir_raw;
						#endif

						// New BRDF path:
						// no legacy kspec multiplier on GGX spec derivatives.

						float dL_dndoth = dL_dspec * dspec_dndoth;
						float dL_dndotl_from_spec = dL_dspec * dspec_dndotl;

						if (dL_dndoth != 0.0f)
						{
							g_unit.x += dL_dndoth * Hh.x;
							g_unit.y += dL_dndoth * Hh.y;
							g_unit.z += dL_dndoth * Hh.z;
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
				float3 g_raw = apply_norm_jacobian(n_raw, g_unit);

				const float gmax = 5.0f;
				g_raw.x = fminf(fmaxf(g_raw.x, -gmax), gmax);
				g_raw.y = fminf(fmaxf(g_raw.y, -gmax), gmax);
				g_raw.z = fminf(fmaxf(g_raw.z, -gmax), gmax);

				atomicAdd(&dL_dnormal3D[global_id * 3 + 0], g_raw.x);
				atomicAdd(&dL_dnormal3D[global_id * 3 + 1], g_raw.y);
				atomicAdd(&dL_dnormal3D[global_id * 3 + 2], g_raw.z);

				// ---------- point_cam lighting gradient approximation ----------
				{
					float3 n_for_geom = n;

					float3 g_point_light = pointcam_lighting_grad_approx(
						Lout,
						point_cam,
						n_for_geom,
						dL_ddiffuse,
						dL_dspec
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
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif

			if (rho3d <= rho2d) {
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

	#if LIGHT_ENABLE_BWD && LIGHT_USE_LAMBERT && (LIGHT_AMBIENT_MODE == 2)
	{
		// write each threads contribution
		amb_reduce[block.thread_rank()] = dAmb;
		block.sync();

		// reduce within block
		for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1)
		{
			if (block.thread_rank() < stride)
				amb_reduce[block.thread_rank()] += amb_reduce[block.thread_rank() + stride];
			block.sync();
		}

		// store one atomic per block
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
	 glm::vec4* dL_drots)
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
	glm::vec4* dL_drots)
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
		dL_drots
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
	glm::vec4* dL_drots)
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
		dL_drots
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
	float* dL_dmetallic)
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
		dL_dmetallic
		);
}