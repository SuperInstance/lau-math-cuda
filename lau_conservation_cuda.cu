/*
 * lau_conservation_cuda.cu — Parallel Noether charge verification
 *
 * CRDT merge via atomics, conservation ratio reduction.
 */

#include "lau_dispatch.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                               \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "[lau_conservation] CUDA error %s:%d: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return -1;                                                  \
        }                                                               \
    } while (0)

/* ── Conservation ratio kernel ─────────────────────────────────── */

__global__ void conservation_ratio_kernel(const double *charges_in,
                                          const double *charges_out,
                                          double *ratios,
                                          int num_agents)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;

    double cin = charges_in[idx];
    if (fabs(cin) < 1e-15) {
        ratios[idx] = 0.0; /* undefined → 0 */
    } else {
        ratios[idx] = charges_out[idx] / cin;
    }
}

/* ── CRDT merge kernel: average via atomic accumulation ────────── */

__global__ void crdt_merge_kernel(const double *local_states,
                                  double *merged,
                                  int state_len, int num_agents)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_len) return;

    double sum = 0.0;
    for (int a = 0; a < num_agents; a++) {
        sum += local_states[(size_t)a * state_len + idx];
    }
    merged[idx] = sum / num_agents;
}

/* ── Reduction kernel: compute mean of ratios ──────────────────── */

__global__ void ratio_reduce_kernel(const double *ratios,
                                    double *global_sum,
                                    int num_agents)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    atomicAdd(global_sum, ratios[idx]);
}

/* ── Public API ────────────────────────────────────────────────── */

int lau_verify_conservation_batch(const double *charges_in,
                                  const double *charges_out,
                                  double *ratios,
                                  int num_agents)
{
    if (num_agents < 1) return -1;

    double *d_cin, *d_cout, *d_ratios;
    size_t bytes = num_agents * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_cin, bytes));
    CUDA_CHECK(cudaMalloc(&d_cout, bytes));
    CUDA_CHECK(cudaMalloc(&d_ratios, bytes));

    CUDA_CHECK(cudaMemcpy(d_cin, charges_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cout, charges_out, bytes, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (num_agents + block - 1) / block;
    conservation_ratio_kernel<<<grid, block>>>(d_cin, d_cout, d_ratios, num_agents);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(ratios, d_ratios, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_cin); cudaFree(d_cout); cudaFree(d_ratios);
    return 0;
}

int lau_crdt_merge(const double *local_states, double *merged_out,
                   int state_len, int num_agents)
{
    if (state_len < 1 || num_agents < 1) return -1;

    size_t total = (size_t)num_agents * state_len * sizeof(double);

    double *d_local, *d_merged;
    CUDA_CHECK(cudaMalloc(&d_local, total));
    CUDA_CHECK(cudaMalloc(&d_merged, state_len * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_local, local_states, total, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (state_len + block - 1) / block;
    crdt_merge_kernel<<<grid, block>>>(d_local, d_merged, state_len, num_agents);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(merged_out, d_merged,
                           state_len * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_local); cudaFree(d_merged);
    return 0;
}

int lau_conservation_reduce(const double *ratios, double *global_ratio,
                            int num_agents)
{
    if (num_agents < 1) return -1;

    double *d_ratios, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_ratios, num_agents * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_ratios, ratios, num_agents * sizeof(double),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));

    int block = 256;
    int grid = (num_agents + block - 1) / block;
    ratio_reduce_kernel<<<grid, block>>>(d_ratios, d_sum, num_agents);
    CUDA_CHECK(cudaGetLastError());

    double h_sum;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    *global_ratio = h_sum / num_agents;

    cudaFree(d_ratios); cudaFree(d_sum);
    return 0;
}
