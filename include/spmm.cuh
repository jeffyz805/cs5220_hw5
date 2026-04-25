#ifndef SPMM_CUH
#define SPMM_CUH
#include "common.h"
#include "CSR.hpp"

namespace spmm {

__device__ __forceinline__ int find_row(const uint32_t * __restrict__ rowptrs,
                                        int m, uint32_t target)
{
    int lo = 0, hi = m;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (__ldg(&rowptrs[mid]) <= target) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

template <int TILE>
__global__ void SpMM_merge(const int m, const int k,
                           const float * __restrict__ d_A_vals,
                           const uint32_t * __restrict__ d_A_colinds,
                           const uint32_t * __restrict__ d_A_rowptrs,
                           const float * __restrict__ d_X,
                           float * __restrict__ d_Y,
                           const int total_nnz, const int num_warps)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= num_warps) return;
    const int lane = threadIdx.x & 31;

    const int nz_start = (int)((long long)warp_id * total_nnz / num_warps);
    const int nz_end   = (int)((long long)(warp_id + 1) * total_nnz / num_warps);
    if (nz_start >= nz_end) return;

    int row           = find_row(d_A_rowptrs, m, (uint32_t)nz_start);
    const int last_row = find_row(d_A_rowptrs, m, (uint32_t)(nz_end - 1));

    while (row <= last_row) {
        const int rp_start  = (int)__ldg(&d_A_rowptrs[row]);
        const int rp_end    = (int)__ldg(&d_A_rowptrs[row + 1]);
        const int my_nz_beg = max(rp_start, nz_start);
        const int my_nz_end = min(rp_end, nz_end);

        if (my_nz_beg >= my_nz_end) { row++; continue; }


        float acc[TILE];
        #pragma unroll
        for (int t = 0; t < TILE; t++) acc[t] = 0.0f;
        for (int chunk = my_nz_beg; chunk < my_nz_end; chunk += 32) {
            const int nz_idx = chunk + lane;
            const bool valid = (nz_idx < my_nz_end);
            float A_val = valid ? __ldg(&d_A_vals[nz_idx]) : 0.0f;
            int   col   = valid ? (int)__ldg(&d_A_colinds[nz_idx]) : 0;

            const int valid_count = min(32, my_nz_end - chunk);

            for (int i = 0; i < valid_count; i++) {
                const float bA = __shfl_sync(0xffffffff, A_val, i);
                const int   bC = __shfl_sync(0xffffffff, col, i);

                const float *xr = d_X + (size_t)bC * k;

                #pragma unroll
                for (int t = 0; t < TILE; t++) {
                    const int c = t * 32 + lane;
                    if (c < k) {
                        acc[t] += bA * __ldg(&xr[c]);
                    }
                }
            }
        }

        const bool fully_owned = (rp_start >= nz_start) && (rp_end <= nz_end);
        float *yr = d_Y + (size_t)row * k;

        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            const int c = t * 32 + lane;
            if (c < k) {
                if (fully_owned) yr[c] = acc[t];
                else atomicAdd(&yr[c], acc[t]);
            }
        }

        row++;
    }
}

template <int TILE4>
__global__ void SpMM_merge_vec4(const int m, const int k,
                                const float * __restrict__ d_A_vals,
                                const uint32_t * __restrict__ d_A_colinds,
                                const uint32_t * __restrict__ d_A_rowptrs,
                                const float * __restrict__ d_X,
                                float * __restrict__ d_Y,
                                const int total_nnz, const int num_warps)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= num_warps) return;
    const int lane = threadIdx.x & 31;

    const int nz_start = (int)((long long)warp_id * total_nnz / num_warps);
    const int nz_end   = (int)((long long)(warp_id + 1) * total_nnz / num_warps);
    if (nz_start >= nz_end) return;

    int row            = find_row(d_A_rowptrs, m, (uint32_t)nz_start);
    const int last_row = find_row(d_A_rowptrs, m, (uint32_t)(nz_end - 1));

    const int k4 = k >> 2;
    const float4 *X4 = reinterpret_cast<const float4 *>(d_X);
    float4       *Y4 = reinterpret_cast<float4 *>(d_Y);

    while (row <= last_row) {
        const int rp_start  = (int)__ldg(&d_A_rowptrs[row]);
        const int rp_end    = (int)__ldg(&d_A_rowptrs[row + 1]);
        const int my_nz_beg = max(rp_start, nz_start);
        const int my_nz_end = min(rp_end, nz_end);

        if (my_nz_beg >= my_nz_end) { row++; continue; }

        float4 acc[TILE4];
        #pragma unroll
        for (int t = 0; t < TILE4; t++)
            acc[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int chunk = my_nz_beg; chunk < my_nz_end; chunk += 32) {
            const int nz_idx = chunk + lane;
            const bool valid = (nz_idx < my_nz_end);
            float A_val = valid ? __ldg(&d_A_vals[nz_idx]) : 0.0f;
            int   col   = valid ? (int)__ldg(&d_A_colinds[nz_idx]) : 0;

            const int valid_count = min(32, my_nz_end - chunk);

            for (int i = 0; i < valid_count; i++) {
                const float bA = __shfl_sync(0xffffffff, A_val, i);
                const int   bC = __shfl_sync(0xffffffff, col, i);

                const float4 *xr = X4 + (size_t)bC * k4;

                #pragma unroll
                for (int t = 0; t < TILE4; t++) {
                    const int idx = t * 32 + lane;
                    if (idx < k4) {
                        float4 xv = __ldg(&xr[idx]);
                        acc[t].x += bA * xv.x;
                        acc[t].y += bA * xv.y;
                        acc[t].z += bA * xv.z;
                        acc[t].w += bA * xv.w;
                    }
                }
            }
        }

        const bool fully_owned = (rp_start >= nz_start) && (rp_end <= nz_end);

        if (fully_owned) {
            float4 *yr = Y4 + (size_t)row * k4;
            #pragma unroll
            for (int t = 0; t < TILE4; t++) {
                const int idx = t * 32 + lane;
                if (idx < k4) yr[idx] = acc[t];
            }
        } else {
            float *yr = d_Y + (size_t)row * k;
            #pragma unroll
            for (int t = 0; t < TILE4; t++) {
                const int base = (t * 32 + lane) << 2;
                if (base + 3 < k) {
                    atomicAdd(&yr[base],     acc[t].x);
                    atomicAdd(&yr[base + 1], acc[t].y);
                    atomicAdd(&yr[base + 2], acc[t].z);
                    atomicAdd(&yr[base + 3], acc[t].w);
                }
            }
        }

        row++;
    }
}

void SpMM_wrapper(csr_t& A, float * d_X, float * d_Y, const size_t k)
{
    const int m = (int)A.get_rows();

    uint32_t total_nnz_h;
    cudaMemcpy(&total_nnz_h, A.get_rowptrs() + m,
               sizeof(uint32_t), cudaMemcpyDeviceToHost);
    const int total_nnz = (int)total_nnz_h;

    cudaMemset(d_Y, 0, (size_t)m * k * sizeof(float));

    const int nnz_per_warp = 256;
    int num_warps = max(1, (total_nnz + nnz_per_warp - 1) / nnz_per_warp);

    const int tpb = 256;
    const int blocks = (num_warps * 32 + tpb - 1) / tpb;

    if (k % 128 == 0 && k >= 128) {
        const int tile4 = (int)k / 128;
        switch (tile4) {
            case 1:
                SpMM_merge_vec4<1><<<blocks, tpb>>>(m, (int)k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                    d_X, d_Y, total_nnz, num_warps);
                break;
            case 2:
                SpMM_merge_vec4<2><<<blocks, tpb>>>(m, (int)k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                    d_X, d_Y, total_nnz, num_warps);
                break;
            default:
                SpMM_merge_vec4<2><<<blocks, tpb>>>(m, (int)k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                    d_X, d_Y, total_nnz, num_warps);
                break;
        }
    } else {
        // Scalar path
        const int tile = ((int)k + 31) / 32;
        switch (tile) {
            case 1:
                SpMM_merge<1><<<blocks, tpb>>>(m, (int)k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                    d_X, d_Y, total_nnz, num_warps);
                break;
            case 2:
                SpMM_merge<2><<<blocks, tpb>>>(m, (int)k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                    d_X, d_Y, total_nnz, num_warps);
                break;
            default:
                SpMM_merge<8><<<blocks, tpb>>>(m, (int)k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                    d_X, d_Y, total_nnz, num_warps);
                break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
