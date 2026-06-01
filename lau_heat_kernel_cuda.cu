/*
 * lau_heat_kernel_cuda.cu — Parallel heat kernel e^{-tL} computation
 *
 * Batched across agents/timesteps, memory-efficient with buffer reuse.
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
            fprintf(stderr, "[lau_heat] CUDA error %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return -1;                                                  \
        }                                                               \
    } while (0)

/* ── CSR SpMV kernel ───────────────────────────────────────────── */

__global__ void spmv_kernel(const int *row_ptr, const int *col_idx,
                            const double *vals, const double *x,
                            double *y, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    double sum = 0.0;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
        sum += vals[j] * x[col_idx[j]];
    }
    y[row] = sum;
}

/* ── Scale and add kernel: y = a*x + b*y ───────────────────────── */

__global__ void axpby_kernel(const double *x, double *y,
                             double a, double b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    y[idx] = a * x[idx] + b * y[idx];
}

/* ── Copy kernel ───────────────────────────────────────────────── */

__global__ void copy_kernel(const double *src, double *dst, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = src[idx];
}

/* ── Shared workspace for buffer reuse ─────────────────────────── */

typedef struct {
    double *d_x;       /* current vector [N] */
    double *d_y;       /* result of L*x [N] */
    double *d_result;  /* accumulator [N] */
    double *d_temp;    /* temp [N] */
    int *d_row_ptr;
    int *d_col_idx;
    double *d_vals;
    int N;
    int nnz;
    int initialized;
} HeatKernelWorkspace;

static HeatKernelWorkspace g_workspace = {0};

static int hk_init(const int *row_ptr, const int *col_idx,
                   const double *vals, int N, int nnz)
{
    if (g_workspace.initialized && g_workspace.N == N) return 0;

    if (g_workspace.initialized) {
        cudaFree(g_workspace.d_x); cudaFree(g_workspace.d_y);
        cudaFree(g_workspace.d_result); cudaFree(g_workspace.d_temp);
        cudaFree(g_workspace.d_row_ptr); cudaFree(g_workspace.d_col_idx);
        cudaFree(g_workspace.d_vals);
    }

    g_workspace.N = N;
    g_workspace.nnz = nnz;

    CUDA_CHECK(cudaMalloc(&g_workspace.d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_workspace.d_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_workspace.d_result, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_workspace.d_temp, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_workspace.d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_workspace.d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_workspace.d_vals, nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(g_workspace.d_row_ptr, row_ptr,
                           (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_workspace.d_col_idx, col_idx,
                           nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_workspace.d_vals, vals,
                           nnz * sizeof(double), cudaMemcpyHostToDevice));

    g_workspace.initialized = 1;
    return 0;
}

static void hk_teardown(void)
{
    if (!g_workspace.initialized) return;
    cudaFree(g_workspace.d_x); cudaFree(g_workspace.d_y);
    cudaFree(g_workspace.d_result); cudaFree(g_workspace.d_temp);
    cudaFree(g_workspace.d_row_ptr); cudaFree(g_workspace.d_col_idx);
    cudaFree(g_workspace.d_vals);
    g_workspace.initialized = 0;
}

/*
 * Heat kernel via Taylor series: e^{-tL} x ≈ Σ (-t)^k / k! * L^k * x
 * Chebyshev approximation would be more accurate; Taylor is simpler for now.
 * Uses K=20 terms which gives good accuracy for moderate t.
 */
#define HEAT_KERNEL_TAYLOR_ORDER 20

static int compute_heat_single(const double *h_x, double *h_y,
                               double t, int N)
{
    HeatKernelWorkspace *ws = &g_workspace;
    int block = 256;
    int grid_n = (N + block - 1) / block;

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(ws->d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

    /* result = x (k=0 term) */
    copy_kernel<<<grid_n, block>>>(ws->d_x, ws->d_result, N);

    /* temp = x, then iteratively: temp = -t * L * temp */
    copy_kernel<<<grid_n, block>>>(ws->d_x, ws->d_temp, N);

    double coeff = -t;
    for (int k = 1; k <= HEAT_KERNEL_TAYLOR_ORDER; k++) {
        /* y = L * temp */
        spmv_kernel<<<grid_n, block>>>(ws->d_row_ptr, ws->d_col_idx,
                                        ws->d_vals, ws->d_temp, ws->d_y, N);
        CUDA_CHECK(cudaGetLastError());

        /* temp = coeff * y  (which is -t * L * previous_temp) */
        copy_kernel<<<grid_n, block>>>(ws->d_y, ws->d_temp, N);
        scale_kernel_inline(ws->d_temp, coeff, N, grid_n, block);

        /* result += (1/k!) * temp */
        double factorial_k = 1.0;
        for (int j = 2; j <= k; j++) factorial_k *= j;
        double add_coeff = 1.0 / factorial_k;

        /* result += add_coeff * temp */
        axpby_kernel<<<grid_n, block>>>(ws->d_temp, ws->d_result,
                                         add_coeff, 1.0, N);

        coeff *= -t; /* accumulate (-t)^k */
    }

    CUDA_CHECK(cudaMemcpy(h_y, ws->d_result, N * sizeof(double),
                           cudaMemcpyDeviceToHost));
    return 0;
}

/* Inline scale to avoid another kernel */
static __device__ void scale_dev(double *x, double s, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) x[idx] *= s;
}

/* We'll just use axpby with b=1, a=scale */
static void scale_kernel_inline(double *d_x, double scale, int N,
                                int grid_n, int block)
{
    double ones_h = 0.0; /* dummy */
    /* Use a trick: set d_y = d_x, then axpby(d_x, d_x, scale, 0, N) */
    axpby_kernel<<<grid_n, block>>>(d_x, d_x, 0.0, scale, N);
}

/* ── Public API ────────────────────────────────────────────────── */

int lau_heat_kernel_batched(const int *row_ptr, const int *col_idx,
                            const double *vals, int N, int nnz,
                            const double *x, double *y,
                            const double *t_values, int batch_size)
{
    int rc = hk_init(row_ptr, col_idx, vals, N, nnz);
    if (rc != 0) return rc;

    for (int b = 0; b < batch_size; b++) {
        rc = compute_heat_single(x + (size_t)b * N,
                                 y + (size_t)b * N,
                                 t_values[b], N);
        if (rc != 0) return rc;
    }

    hk_teardown();
    return 0;
}

int lau_heat_kernel_multi_source(const int *row_ptr, const int *col_idx,
                                 const double *vals, int N, int nnz,
                                 const double *sources, double *results,
                                 double t, int num_sources)
{
    int rc = hk_init(row_ptr, col_idx, vals, N, nnz);
    if (rc != 0) return rc;

    for (int s = 0; s < num_sources; s++) {
        rc = compute_heat_single(sources + (size_t)s * N,
                                 results + (size_t)s * N,
                                 t, N);
        if (rc != 0) return rc;
    }

    hk_teardown();
    return 0;
}
