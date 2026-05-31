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

#include "forward.h"
#include "auxiliary.h"
#include "lighting.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal,
	float3& bu_cam,
	float3& bv_cam
) {
	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	float3 p_view = transformPoint4x3(p_orig, viewmatrix);

	bu_cam = transformVec4x3(
		make_float3(L[0].x, L[0].y, L[0].z),
		viewmatrix
	);

	bv_cam = transformVec4x3(
		make_float3(L[1].x, L[1].y, L[1].z),
		viewmatrix
	);

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
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

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;

	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float3* means3D_cam,
	float3* basis_u_cam,
    float3* basis_v_cam,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	float3 bu_cam = make_float3(0.0f, 0.0f, 0.0f);
	float3 bv_cam = make_float3(0.0f, 0.0f, 0.0f);

	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal, bu_cam, bv_cam);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);

		bu_cam = make_float3(0.0f, 0.0f, 0.0f);
		bv_cam = make_float3(0.0f, 0.0f, 0.0f);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	means3D_cam[idx] = p_view;
	basis_u_cam[idx] = bu_cam;
	basis_v_cam[idx] = bv_cam;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

__device__ __forceinline__
void set_debug_gray(float C[3], float v)
{
    v = saturate01(v);
    C[0] = v;
    C[1] = v;
    C[2] = v;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ ambients,
	const float* __restrict__ intensity,
	const float* __restrict__ roughness,
	const float* __restrict__ metallic,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	const float3* __restrict__ means3D_cam,
	const float3* __restrict__ basis_u_cam,
    const float3* __restrict__ basis_v_cam,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	__shared__ float3 collected_center_cam[BLOCK_SIZE];
	__shared__ float3 collected_basis_u_cam[BLOCK_SIZE];
	__shared__ float3 collected_basis_v_cam[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

#if LIGHT_SURFACE_SHADING_MODE

	float surf_w_sum = 0.0f;

	float3 surf_P_sum = make_float3(0.0f, 0.0f, 0.0f);
	float3 surf_N_sum = make_float3(0.0f, 0.0f, 0.0f);
	float3 surf_base_sum = make_float3(0.0f, 0.0f, 0.0f);

	float surf_rough_sum = 0.0f;
	float surf_metal_sum = 0.0f;
	float surf_depth_sum = 0.0f;

#endif

#if (LIGHT_DEBUG_MODE > 0)
	float dbg_best = 0.0f;
	float dbg_best_w = 0.0f;

	float dbg_sum = 0.0f;
	float dbg_sum_w = 0.0f;
#endif

	float metallic_accum  = 0.0f;
	float roughness_accum = 0.0f;
	
#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			
			collected_center_cam[block.thread_rank()] = means3D_cam[coll_id];
			collected_basis_u_cam[block.thread_rank()] = basis_u_cam[coll_id];
    		collected_basis_v_cam[block.thread_rank()] = basis_v_cam[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];

			// NEW: Gaussian center in camera space for stable falloff
			const float3 center_cam = collected_center_cam[j];
			const float3 bu_cam = collected_basis_u_cam[j];
			const float3 bv_cam = collected_basis_v_cam[j];

			// Transform the two planes into local u-v system. 
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			// Cross product of two planes is a line, Eq. (9)
			float3 p = cross(k, l);
			if (fabsf(p.z) < 1e-8f) continue;
			// Perspective division to get the intersection (u,v), Eq. (10)
						float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 

			// Add low pass filter
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
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			bool depth_valid = true;

			if (!use_3d_footprint)
			{
				point_cam = center_cam;

				#if LIGHT_DEPTH_DISCARD_2D_FALLBACK
					depth_valid = false;
					depth = Tw.z; // keep harmless fallback for lighting/debug if needed
				#else
					depth = Tw.z;
				#endif
			}

			if (depth_valid && depth < near_n)
				continue;

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < LIGHT_ALPHA_SKIP_THRESHOLD)
				continue;

			float test_T = T * (1 - alpha);

			// ================= LAMBERT + PHONG SHADING (FORWARD) ======================
			float w = alpha * T;

			{
				const int gid_mat = collected_id[j];

				LightMaterialValues mat = eval_light_material_values(
					metallic != nullptr ? metallic + gid_mat : nullptr,
					roughness != nullptr ? roughness + gid_mat : nullptr
				);

				float m_val = mat.metallic;
				float r_val = mat.roughness;

				// Correct visible contribution weight.
				// alpha alone ignores occlusion; alpha*T matches color compositing.
				const float material_w = w;

				metallic_accum  += w * m_val;
				roughness_accum += w * r_val;
			}

			float w_indirect = 0.0f;
			float w_direct   = w;

			LightingOut Lout = {};

			#if !LIGHT_SURFACE_SHADING_MODE
			const float* rough_ptr = nullptr;
			const float* metal_ptr = nullptr;

			#if LIGHT_ENABLE_FWD && (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
			{
				float3 n_raw = make_float3(normal[0], normal[1], normal[2]);
				const int gid = collected_id[j];

				rough_ptr = roughness + gid;
				metal_ptr = metallic + gid;

				float3 base_rgb = make_float3(
					features[gid * CHANNELS + 0],
					features[gid * CHANNELS + 1],
					features[gid * CHANNELS + 2]
				);

				Lout = eval_lighting(
					pixf, W, H, focal_x, focal_y,
					n_raw, depth,
					ambients, intensity,
					rough_ptr, metal_ptr,
					base_rgb,
					&bu_cam,
					&bv_cam,
					&point_cam
				);

				w_indirect = w * Lout.indirect_diffuse;
				w_direct   = w * Lout.direct_diffuse;
			}
			#else
			{
				w_indirect = 0.0f;
				w_direct   = w;
			}
			#endif
			#endif

#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			if (depth_valid)
			{
				float A = 1 - T;
				float m = far_n / (far_n - near_n) * (1 - near_n / depth);

				distortion += (m * m * A + M2 - 2 * m * M1) * w;
				D  += depth * w;
				M1 += m * w;
				M2 += m * m * w;

				if (T > 0.5f)
				{
					median_depth = depth;
					// median_weight = w;
					median_contributor = contributor;
				}
			}

			// Render normal map
			if (depth_valid)
			{
				for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
			}
#endif

			#if (LIGHT_DEBUG_MODE > 0) && !LIGHT_SURFACE_SHADING_MODE
			{
				float dbg = 0.0f;

				// useful geometric helpers for debug visualization
				const float dx_pc = point_cam.x - center_cam.x;
				const float dy_pc = point_cam.y - center_cam.y;
				const float dz_pc = point_cam.z - center_cam.z;
				const float point_disp = sqrtf(dx_pc * dx_pc + dy_pc * dy_pc + dz_pc * dz_pc);

				const float3 light_pos_dbg = make_float3(0.0f, 0.0f, 0.0f);
				const float lx_dbg = point_cam.x - light_pos_dbg.x;
				const float ly_dbg = point_cam.y - light_pos_dbg.y;
				const float lz_dbg = point_cam.z - light_pos_dbg.z;
				const float dist_dbg = sqrtf(lx_dbg * lx_dbg + ly_dbg * ly_dbg + lz_dbg * lz_dbg);

				#if (LIGHT_DEBUG_MODE == 1)
					// distance from chosen lighting point to light
					dbg = 1.0f / (1.0f + dist_dbg);

				#elif (LIGHT_DEBUG_MODE == 2)
					// falloff inv only
					dbg = Lout.inv / (1.0f + Lout.inv);

				#elif (LIGHT_DEBUG_MODE == 3)
					// final lighting intensity after I * inv
					dbg = Lout.intensity / (1.0f + Lout.intensity);

				#elif (LIGHT_DEBUG_MODE == 4)
					// spotlight factor
					dbg = Lout.spot * LIGHT_DEBUG_SCALE;

				#elif (LIGHT_DEBUG_MODE == 6)
					// displacement of point_cam from center_cam
					dbg = point_disp / (0.1f + point_disp);

				#elif (LIGHT_DEBUG_MODE == 7)
					// raw normal visualization
					C[0] = 0.5f * (normal[0] + 1.0f);
					C[1] = 0.5f * (normal[1] + 1.0f);
					C[2] = 0.5f * (normal[2] + 1.0f);

				#elif (LIGHT_DEBUG_MODE == 8)
					// learned ambient
					dbg = Lout.ambient * LIGHT_DEBUG_SCALE;

				#elif (LIGHT_DEBUG_MODE == 9)
					// learned intensity value
					{
						float dI_dummy = 0.0f;
						float I_dbg = intensity_value(intensity, &dI_dummy);
						dbg = I_dbg / (1.0f + I_dbg);
					}

				#elif (LIGHT_DEBUG_MODE == 10)
				{
					const int gid_dbg = collected_id[j];

					#if (LIGHT_GGX_METALLIC_MODE == 1)
						if (metallic != nullptr)
						{
							float dmetal_dummy = 0.0f;
							dbg = metallic_value(metallic + gid_dbg, &dmetal_dummy);
						}
						else
						{
							dbg = 0.0f;
						}
					#else
						float dmetal_dummy = 0.0f;
						dbg = metallic_value(nullptr, &dmetal_dummy);
					#endif
				}

				#elif (LIGHT_DEBUG_MODE == 11)
				{
					// Final per-contribution roughness value.
					// Later blended by dbg_sum += w * dbg.
					const int gid_dbg = collected_id[j];

					#if (LIGHT_GGX_ROUGHNESS_MODE == 1)
						if (roughness != nullptr)
						{
							float drough_dummy = 0.0f;
							dbg = roughness_value(roughness + gid_dbg, &drough_dummy);
						}
						else
						{
							dbg = 0.0f;
						}
					#else
						float drough_dummy = 0.0f;
						dbg = roughness_value(nullptr, &drough_dummy);
					#endif
				}

				#elif (LIGHT_DEBUG_MODE == 12)
					// ndotl
					dbg = 0.5f * (Lout.ndotl + 1.0f);

				#elif (LIGHT_DEBUG_MODE == 13)
					// lambert term
					dbg = Lout.lambert * LIGHT_DEBUG_SCALE;

				#elif (LIGHT_DEBUG_MODE == 14)
					// RGB specular additive contribution shown as scalar proxy
					dbg = Lout.spec_add / (1.0f + Lout.spec_add);

				#elif (LIGHT_DEBUG_MODE == 15)
					// chosen point depth
					dbg = point_cam.z / (1.0f + point_cam.z);

				#elif (LIGHT_DEBUG_MODE == 16)
					// local alpha contribution
					dbg = alpha * 32.0f;

				#elif (LIGHT_DEBUG_MODE == 17)
					// local compositing weight w = alpha * T
					dbg = w * 32.0f;
				#endif

				#if (LIGHT_DEBUG_MODE != 7)
					dbg = saturate01(dbg);

					float dbg_w = w;

					dbg_sum   += dbg_w * dbg;
					dbg_sum_w += dbg_w;

					if (dbg_w > dbg_best_w)
					{
						dbg_best_w = dbg_w;
						dbg_best   = dbg;
					}
				#endif

				// in debug mode, still advance compositing state so the viewer updates correctly
				T = test_T;
				last_contributor = contributor;
                if (T < 0.0001f)
                {
                    done = true;
                }
				continue;
			}
#endif

		#if LIGHT_SURFACE_SHADING_MODE

			{
				const int gid_surf = collected_id[j];

				float3 base_rgb = make_float3(
					features[gid_surf * CHANNELS + 0],
					features[gid_surf * CHANNELS + 1],
					features[gid_surf * CHANNELS + 2]
				);

				float3 n_basis = faceforward_basis_normal(bu_cam, bv_cam, point_cam);

				LightMaterialValues mat = eval_light_material_values(
					metallic != nullptr ? metallic + gid_surf : nullptr,
					roughness != nullptr ? roughness + gid_surf : nullptr
				);

				float m_val = mat.metallic;
				float r_val = mat.roughness;

				surf_w_sum += w;

				surf_P_sum.x += w * point_cam.x;
				surf_P_sum.y += w * point_cam.y;
				surf_P_sum.z += w * point_cam.z;

				surf_N_sum.x += w * n_basis.x;
				surf_N_sum.y += w * n_basis.y;
				surf_N_sum.z += w * n_basis.z;

				surf_base_sum.x += w * base_rgb.x;
				surf_base_sum.y += w * base_rgb.y;
				surf_base_sum.z += w * base_rgb.z;

				surf_rough_sum += w * r_val;
				surf_metal_sum += w * m_val;
				
				// depth may be invalid in fallback mode.
				// Use point_cam.z as the safe fallback depth.
				surf_depth_sum += w * (depth_valid ? depth : point_cam.z);
			}

		#else
			// Diffuse
			#pragma unroll
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				const float albedo = features[collected_id[j] * CHANNELS + ch];

				#if LIGHT_ENABLE_FWD && (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
				if (ch < 3)
				{
					C[ch] += albedo * w * ((&Lout.diffuse_mul_rgb.x)[ch]);

					#if LIGHT_USE_PHONG
					C[ch] += w * ((&Lout.spec_add_rgb.x)[ch]);
					#endif
				}
				else
				{
					C[ch] += albedo * w_direct;
				}
				#else
				C[ch] += albedo * w_direct;
				#endif
			}

		#endif

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

            if (T < 0.0001f)
            {
                done = true;
            }
		}
	}

	#if LIGHT_SURFACE_SHADING_MODE

		if (inside && surf_w_sum > 1e-8f)
		{
			const float invW = 1.0f / surf_w_sum;

			float3 surf_P = make_float3(
				surf_P_sum.x * invW,
				surf_P_sum.y * invW,
				surf_P_sum.z * invW
			);

			float3 surf_N = normalize_or_default(
				make_float3(
					surf_N_sum.x * invW,
					surf_N_sum.y * invW,
					surf_N_sum.z * invW
				),
				make_float3(0.0f, 0.0f, 1.0f)
			);

			float3 surf_base = make_float3(
				surf_base_sum.x * invW,
				surf_base_sum.y * invW,
				surf_base_sum.z * invW
			);

			float surf_rough = surf_rough_sum * invW;
			float surf_metal = surf_metal_sum * invW;
			float surf_depth = surf_depth_sum * invW;

			LightingOut Lsurf =
				eval_lighting_surface_values(
					pixf,
					W, H,
					focal_x,
					focal_y,
					surf_N,
					surf_depth,
					ambients,
					intensity,
					surf_rough,
					surf_metal,
					surf_base,
					&surf_P
				);

			const float surface_alpha = 1.0f - T;

			C[0] = surface_alpha * (
				surf_base.x * Lsurf.diffuse_mul_rgb.x +
				Lsurf.spec_add_rgb.x
			);

			C[1] = surface_alpha * (
				surf_base.y * Lsurf.diffuse_mul_rgb.y +
				Lsurf.spec_add_rgb.y
			);

			C[2] = surface_alpha * (
				surf_base.z * Lsurf.diffuse_mul_rgb.z +
				Lsurf.spec_add_rgb.z
			);

			for (int ch = 3; ch < CHANNELS; ch++)
				C[ch] = 0.0f;
		}
	#endif

#if (LIGHT_DEBUG_MODE > 0) && (LIGHT_DEBUG_MODE != 7)
if (inside)
{
    float dbg_final = (dbg_sum_w > 1e-8f)
        ? (dbg_sum / dbg_sum_w)
        : dbg_best;

    dbg_final = saturate01(dbg_final);

    #if (LIGHT_DEBUG_MODE == 10 || LIGHT_DEBUG_MODE == 11)
        dbg_final = floorf(dbg_final * 10.0f + 0.5f) / 10.0f;

		// approximate value bands:
		// 0.0 black
		// 0.1 blue
		// 0.2 cyan
		// 0.3 green
		// 0.4 yellow-green
		// 0.5 yellow
		// 0.6 orange
		// 0.7 red-orange
		// 0.8 red
		// 0.9 magenta
		// 1.0 white
        if      (dbg_final < 0.05f) { C[0]=0.0f; C[1]=0.0f; C[2]=0.0f; } // 0.0
        else if (dbg_final < 0.15f) { C[0]=0.0f; C[1]=0.0f; C[2]=1.0f; } // 0.1
        else if (dbg_final < 0.25f) { C[0]=0.0f; C[1]=1.0f; C[2]=1.0f; } // 0.2
        else if (dbg_final < 0.35f) { C[0]=0.0f; C[1]=1.0f; C[2]=0.0f; } // 0.3
        else if (dbg_final < 0.45f) { C[0]=0.5f; C[1]=1.0f; C[2]=0.0f; } // 0.4
        else if (dbg_final < 0.55f) { C[0]=1.0f; C[1]=1.0f; C[2]=0.0f; } // 0.5
        else if (dbg_final < 0.65f) { C[0]=1.0f; C[1]=0.5f; C[2]=0.0f; } // 0.6
        else if (dbg_final < 0.75f) { C[0]=1.0f; C[1]=0.25f; C[2]=0.0f; } // 0.7
        else if (dbg_final < 0.85f) { C[0]=1.0f; C[1]=0.0f; C[2]=0.0f; } // 0.8
        else if (dbg_final < 0.95f) { C[0]=1.0f; C[1]=0.0f; C[2]=1.0f; } // 0.9
        else                        { C[0]=1.0f; C[1]=1.0f; C[2]=1.0f; } // 1.0

    #else
        C[0] = dbg_final;
        C[1] = dbg_final;
        C[2] = dbg_final;
    #endif
}
#endif

	const float final_alpha = 1.0f - T;

	float metallic_final  = 0.0f;
	float roughness_final = 0.0f;

	#if LIGHT_SURFACE_SHADING_MODE
	if (surf_w_sum > 1e-8f)
	{
		const float invW = 1.0f / surf_w_sum;
		metallic_final  = surf_metal_sum * invW;
		roughness_final = surf_rough_sum * invW;
	}
	#else
	if (final_alpha > 1e-8f)
	{
		metallic_final  = metallic_accum  / final_alpha;
		roughness_final = roughness_accum / final_alpha;
	}
	#endif

	metallic_final  = saturate01(metallic_final);
	roughness_final = saturate01(roughness_final);

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		#if (LIGHT_DEBUG_MODE > 0)
		out_color[0 * H * W + pix_id] = C[0];
		out_color[1 * H * W + pix_id] = C[1];
		out_color[2 * H * W + pix_id] = C[2];
		#else
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		#endif

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + METALLIC_OFFSET * H * W] = metallic_final;
		out_others[pix_id + ROUGHNESS_OFFSET * H * W] = roughness_final;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* ambients,
	const float* intensity,
	const float* roughness,
	const float* metallic,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	const float3* means3D_cam,
	const float3* basis_u_cam,
    const float3* basis_v_cam,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		ambients,
		intensity,
		roughness,
		metallic,
		transMats,
		depths,
		normal_opacity,
		means3D_cam,
		basis_u_cam,
		basis_v_cam,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float3* means3D_cam,
	float3* basis_u_cam,
    float3* basis_v_cam,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		means3D_cam,
		basis_u_cam,
		basis_v_cam,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
