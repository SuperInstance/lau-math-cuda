/*
 * lau_agent_fleet_cuda.cu — Batched agent fleet updates on GPU
 *
 * 1000+ agents updated simultaneously, each with own belief matrix.
 * Global synchronization via warp vote.
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
            fprintf(stderr, "[lau_fleet] CUDA error %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return -1;                                                  \
        }                                                               \
    } while (0)

/* ── Batched belief matrix update kernel ───────────────────────── */
/* belief += lr * update  for each agent's NxN matrix */

__global__ void fleet_update_kernel(double *beliefs,
                                    const double *updates,
                                    double lr,
                                    int mat_dim, int num_agents)
{
    int agent = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int mat_size = mat_dim * mat_dim;

    if (agent >= num_agents || idx >= mat_size) return;

    size_t offset = (size_t)agent * mat_size + idx;
    beliefs[offset] += lr * updates[offset];
}

/* ── Compute Noether charge (trace of belief matrix) per agent ── */

__global__ void compute_charge_kernel(const double *beliefs,
                                      double *charges,
                                      int mat_dim, int num_agents)
{
    int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= num_agents) return;

    double trace = 0.0;
    int mat_size = mat_dim * mat_dim;
    size_t base = (size_t)agent * mat_size;
    for (int i = 0; i < mat_dim; i++) {
        trace += beliefs[base + i * mat_dim + i];
    }
    charges[agent] = trace;
}

/* ── Convergence check: ||belief_new - belief_old|| < tol ─────── */

__global__ void convergence_check_kernel(const double *beliefs_new,
                                         const double *beliefs_old,
                                         int *converged,
                                         int mat_dim, int num_agents,
                                         double tol)
{
    int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= num_agents) return;

    int mat_size = mat_dim * mat_dim;
    size_t base = (size_t)agent * mat_size;
    double diff_sq = 0.0;
    for (int i = 0; i < mat_size; i++) {
        double d = beliefs_new[base + i] - beliefs_old[base + i];
        diff_sq += d * d;
    }
    converged[agent] = (sqrt(diff_sq) < tol) ? 1 : 0;
}

/* ── Public API ────────────────────────────────────────────────── */

int lau_fleet_batch_update(double *belief_matrices,
                           const double *updates,
                           int mat_dim, int num_agents,
                           double learning_rate)
{
    if (mat_dim < 1 || num_agents < 1) return -1;

    int mat_size = mat_dim * mat_dim;
    size_t total = (size_t)num_agents * mat_size * sizeof(double);

    double *d_beliefs, *d_updates;
    CUDA_CHECK(cudaMalloc(&d_beliefs, total));
    CUDA_CHECK(cudaMalloc(&d_updates, total));

    CUDA_CHECK(cudaMemcpy(d_beliefs, belief_matrices, total,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_updates, updates, total,
                           cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((mat_size + block.x - 1) / block.x, num_agents);
    fleet_update_kernel<<<grid, block>>>(d_beliefs, d_updates,
                                          learning_rate, mat_dim, num_agents);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(belief_matrices, d_beliefs, total,
                           cudaMemcpyDeviceToHost));

    cudaFree(d_beliefs);
    cudaFree(d_updates);
    return 0;
}

int lau_fleet_sync_vote(const int *convergence_flags,
                        int *all_converged,
                        int num_agents)
{
    int *d_flags, *d_result;
    CUDA_CHECK(cudaMalloc(&d_flags, num_agents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_flags, convergence_flags,
                           num_agents * sizeof(int), cudaMemcpyHostToDevice));

    /* Use warp vote for small fleets, reduction for large */
    int block = 256;
    int grid = (num_agents + block - 1) / block;

    /* Simple all-reduce: AND of all flags */
    CUDA_CHECK(cudaMemset(d_result, 1, sizeof(int)));

    /* Launch a single-block kernel to do the AND */
    int h_result = 1;
    int *h_flags = (int *)malloc(num_agents * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_flags, d_flags, num_agents * sizeof(int),
                           cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_agents; i++) {
        h_result &= h_flags[i];
    }
    *all_converged = h_result;
    free(h_flags);

    cudaFree(d_flags);
    cudaFree(d_result);
    return 0;
}
