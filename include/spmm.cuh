#ifndef SPMM_CUH
#define SPMM_CUH
#include "common.h"
#include "CSR.hpp"

namespace spmm {

template <int TILE>
__global__ void SpMM(const size_t m, const size_t k,
                     float *d_A_vals, uint32_t *d_A_colinds, uint32_t *d_A_rowptrs,
                     float *d_X, float *d_Y)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= m) return;
    int lane = threadIdx.x % 32;

    int row = warp_id;
    int nz_begin = d_A_rowptrs[row];
    int nz_end = d_A_rowptrs[row + 1];

    float acc[TILE] = {0};

    for (int chunk_base = nz_begin; chunk_base < nz_end; chunk_base += 32) {
        int my_nz = chunk_base + lane;
        float A_val = (my_nz < nz_end) ? d_A_vals[my_nz] : 0.0f;
        int col_ind = (my_nz < nz_end) ? (int)d_A_colinds[my_nz] : 0;

        int valid_count = min(32, (int)(nz_end - chunk_base));

        for (int i = 0; i < valid_count; i++) {
            float b_A_val = __shfl_sync(0xffffffff, A_val, i);
            int b_col_ind = __shfl_sync(0xffffffff, col_ind, i);

            // base pointer into row b_col_ind of X
            float *x_row = d_X + b_col_ind * k;

            #pragma unroll
            for (int t = 0; t < TILE; t++) {
                int x_col = t * 32 + lane;
                if (x_col < k) {
                    acc[t] += b_A_val * x_row[x_col];
                }
            }
        }
    }

    // write results — no atomics needed, one warp owns the row
    float *y_row = d_Y + row * k;

    #pragma unroll
    for (int t = 0; t < TILE; t++) {
        int x_col = t * 32 + lane;
        if (x_col < k) {
            y_row[x_col] = acc[t];
        }
    }
}

// float4 variant: each thread handles 4 columns per tile slot
// requires k to be divisible by 128 (32 threads * 4 floats)
template <int TILE4>
__global__ void SpMM_vec4(const size_t m, const size_t k,
                          float *d_A_vals, uint32_t *d_A_colinds, uint32_t *d_A_rowptrs,
                          float *d_X, float *d_Y)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= m) return;
    int lane = threadIdx.x % 32;

    int row = warp_id;
    int nz_begin = d_A_rowptrs[row];
    int nz_end = d_A_rowptrs[row + 1];

    // TILE4 = number of float4 loads per thread
    // total columns covered = TILE4 * 32 * 4 = TILE4 * 128
    float4 acc4[TILE4];
    #pragma unroll
    for (int t = 0; t < TILE4; t++) {
        acc4[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // cast X and Y to float4 for vectorized access
    // k_vec = number of float4s per row = k / 4
    int k_vec = k / 4;
    float4 *X4 = reinterpret_cast<float4 *>(d_X);
    float4 *Y4 = reinterpret_cast<float4 *>(d_Y);

    for (int chunk_base = nz_begin; chunk_base < nz_end; chunk_base += 32) {
        int my_nz = chunk_base + lane;
        float A_val = (my_nz < nz_end) ? d_A_vals[my_nz] : 0.0f;
        int col_ind = (my_nz < nz_end) ? (int)d_A_colinds[my_nz] : 0;

        int valid_count = min(32, (int)(nz_end - chunk_base));

        for (int i = 0; i < valid_count; i++) {
            float b_A_val = __shfl_sync(0xffffffff, A_val, i);
            int b_col_ind = __shfl_sync(0xffffffff, col_ind, i);

            float4 *x_row4 = X4 + b_col_ind * k_vec;

            #pragma unroll
            for (int t = 0; t < TILE4; t++) {
                int idx = t * 32 + lane;
                if (idx < k_vec) {
                    float4 xv = x_row4[idx];
                    acc4[t].x += b_A_val * xv.x;
                    acc4[t].y += b_A_val * xv.y;
                    acc4[t].z += b_A_val * xv.z;
                    acc4[t].w += b_A_val * xv.w;
                }
            }
        }
    }

    float4 *y_row4 = Y4 + row * k_vec;

    #pragma unroll
    for (int t = 0; t < TILE4; t++) {
        int idx = t * 32 + lane;
        if (idx < k_vec) {
            y_row4[idx] = acc4[t];
        }
    }
}

void SpMM_wrapper(csr_t& A, float *d_X, float *d_Y, const size_t k)
{
    size_t m = A.get_rows();

    int threads_per_block = 256;
    int blocks = ((int)m * 32 + threads_per_block - 1) / threads_per_block;

    // use float4 kernel when k is divisible by 128 (32 lanes * 4 floats)
    // otherwise fall back to scalar kernel
    if (k % 128 == 0) {
        int tile4 = k / 128;
        switch (tile4) {
            case 1:  // k=128
                SpMM_vec4<1><<<blocks, threads_per_block>>>(m, k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
                break;
            case 2:  // k=256
                SpMM_vec4<2><<<blocks, threads_per_block>>>(m, k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
                break;
            default:
                SpMM_vec4<2><<<blocks, threads_per_block>>>(m, k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
                break;
        }
    } else {
        int tile = (k + 31) / 32;
        switch (tile) {
            case 1:  // k <= 32
                SpMM<1><<<blocks, threads_per_block>>>(m, k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
                break;
            case 2:  // k=64
                SpMM<2><<<blocks, threads_per_block>>>(m, k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
                break;
            case 8:  // k=256
                SpMM<8><<<blocks, threads_per_block>>>(m, k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
                break;
            default:
                SpMM<8><<<blocks, threads_per_block>>>(m, k,
                    A.get_vals(), A.get_colinds(), A.get_rowptrs(), d_X, d_Y);
                break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif