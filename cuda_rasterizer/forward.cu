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
	const float* __restrict__ kspecular,
	const float* __restrict__ shiny,
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
			float rho = min(rho3d, rho2d);

			bool reliable_hit = (rho3d <= rho2d);

			// clamp local hit coordinates
			const float uv_clamp = LIGHT_HIT_UV_CLAMP;
			float sx = fmaxf(-uv_clamp, fminf(s.x, uv_clamp));
			float sy = fmaxf(-uv_clamp, fminf(s.y, uv_clamp));

			bool hit_was_clamped = (fabsf(s.x - sx) > 1e-6f) || (fabsf(s.y - sy) > 1e-6f);

			float3 hit_cam = make_float3(
				center_cam.x + sx * bu_cam.x + sy * bv_cam.x,
				center_cam.y + sx * bu_cam.y + sy * bv_cam.y,
				center_cam.z + sx * bu_cam.z + sy * bv_cam.z
			);

			// sanity check on hit displacement
			float3 delta_hit = make_float3(
				hit_cam.x - center_cam.x,
				hit_cam.y - center_cam.y,
				hit_cam.z - center_cam.z
			);

			float delta2 = delta_hit.x * delta_hit.x +
						delta_hit.y * delta_hit.y +
						delta_hit.z * delta_hit.z;

			const float hit_delta2_max = LIGHT_HIT_DELTA2_MAX;
			bool sane_hit = (delta2 <= hit_delta2_max);

			float3 point_cam = (reliable_hit && sane_hit) ? hit_cam : center_cam;

			float surface_conf = 1.0f;

			if (!reliable_hit || !sane_hit)
				surface_conf *= LIGHT_CONF_FALLBACK;

			if (hit_was_clamped)
				surface_conf *= LIGHT_CONF_CLAMPED_HIT;
				
			float point_disp = sqrtf(fmaxf(delta2, 1e-12f));
			float disp_conf = 1.0f / (1.0f + LIGHT_CONF_DISP_K * point_disp);
			surface_conf *= disp_conf;
			surface_conf = fmaxf(0.05f, fminf(surface_conf, 1.0f));

			float3 LP_dbg = make_float3(
				point_cam.x - (-0.01f * 0.70710678f),
				point_cam.y - (-0.01f * 0.70710678f),
				point_cam.z - 0.0f
			);

			float dist2_dbg = LP_dbg.x * LP_dbg.x + LP_dbg.y * LP_dbg.y + LP_dbg.z * LP_dbg.z;
			float dist_dbg = sqrtf(fmaxf(dist2_dbg, 1e-12f));

			float disp_x = point_cam.x - center_cam.x;
			float disp_y = point_cam.y - center_cam.y;
			float disp_z = point_cam.z - center_cam.z;

			// compute depth
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			// if a point is too small, its depth is not reliable?
			// depth = (rho3d <= rho2d) ? depth : Tw.z 
			if (depth < near_n) continue;

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
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < LIGHT_ALPHA_SKIP_THRESHOLD)
				continue;

			// weak contributors should shape shading less
			if (alpha < LIGHT_ALPHA_SOFT_REF)
				surface_conf *= LIGHT_CONF_LOW_ALPHA;

			surface_conf = fmaxf(0.05f, fminf(surface_conf, 1.0f));

			float test_T = T * (1 - alpha);

			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// ================= LAMBERT + PHONG SHADING (FORWARD) ======================
			float w      = alpha * T;
			float w_diff = w;        // default, no lighting
			float w_spec = 0.0f;     // default, no spec

			#if LIGHT_ENABLE_FWD && (LIGHT_USE_LAMBERT || LIGHT_USE_PHONG)
				float3 n_raw = make_float3(normal[0], normal[1], normal[2]);
				const int gid = collected_id[j];
				const float* ks_ptr = kspecular + gid;   // per-gaussian
				const float* shi_ptr = shiny + gid;

				LightingOut Lout = eval_lighting(pixf, W, H, focal_x, focal_y, n_raw, depth, ambients, intensity, ks_ptr, shi_ptr, &point_cam, surface_conf, alpha);

				w_diff = w * Lout.diffuse_mul;

				#if LIGHT_USE_PHONG
					w_spec = w * Lout.spec_add;
				#endif
			#endif

#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

#if (LIGHT_DEBUG_MODE > 0)
			{
				float dbg = 0.0f;

				#if (LIGHT_DEBUG_MODE == 1)
					// distance from chosen lighting point to light
					dbg = 1.0f / (1.0f + dist_dbg);

				#elif (LIGHT_DEBUG_MODE == 2)
					// falloff inv only
					dbg = Lout.inv / (1.0f + Lout.inv);

				#elif (LIGHT_DEBUG_MODE == 3)
					// final lighting intensity after I * inv * confidence and clamp
					dbg = Lout.intensity / (1.0f + Lout.intensity);

				#elif (LIGHT_DEBUG_MODE == 4)
					// spotlight factor
					dbg = Lout.spot * LIGHT_DEBUG_SCALE;

				#elif (LIGHT_DEBUG_MODE == 5)
					// hit-state mask:
					// red = fallback
					// green = reliable+sane
					// blue = clamped
					C[0] = (!reliable_hit || !sane_hit) ? 1.0f : 0.0f;
					C[1] = (reliable_hit && sane_hit) ? 1.0f : 0.0f;
					C[2] = hit_was_clamped ? 1.0f : 0.0f;

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
					// learned intensity value as seen by eval_lighting
					{
						float dI_dummy = 0.0f;
						float I_dbg = intensity_value(intensity, &dI_dummy);
						dbg = I_dbg / (1.0f + I_dbg);
					}

				#elif (LIGHT_DEBUG_MODE == 10)
					// learned kspec
					dbg = kspec_value(ks_ptr) * LIGHT_DEBUG_SCALE;

				#elif (LIGHT_DEBUG_MODE == 11)
					// learned shiny
					{
						float dshin_dummy = 0.0f;
						float s_dbg = shininess_value(shi_ptr, &dshin_dummy);
						dbg = logf(1.0f + s_dbg) / logf(1.0f + LIGHT_SHINY_MAX);
					}

				#elif (LIGHT_DEBUG_MODE == 12)
					// ndotl
					dbg = 0.5f * (Lout.ndotl + 1.0f);

				#elif (LIGHT_DEBUG_MODE == 13)
					// lambert term
					dbg = Lout.lambert * LIGHT_DEBUG_SCALE;

				#elif (LIGHT_DEBUG_MODE == 14)
					// specular additive contribution
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

				#if (LIGHT_DEBUG_MODE != 5) && (LIGHT_DEBUG_MODE != 7)
					dbg = saturate01(dbg);
					C[0] = dbg;
					C[1] = dbg;
					C[2] = dbg;
				#endif

				// in debug mode, still advance compositing state so the viewer updates correctly
				T = test_T;
				last_contributor = contributor;
				continue;
			}
#endif

			// Diffuse
			#pragma unroll
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w_diff;
			}

			// Specular added to RGB
			#if LIGHT_USE_PHONG
				C[0] += w_spec;
				C[1] += w_spec;
				C[2] += w_spec;
			#endif

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
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
	const float* kspecular,
	const float* shiny,
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
		kspecular,
		shiny,
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
