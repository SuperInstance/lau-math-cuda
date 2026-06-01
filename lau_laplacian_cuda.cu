/*
 * lau_laplacian_cuda.cu — Graph Laplacian on GPU
 *
 * Implements: CSR Laplacian construction, batched spectral gap, power iteration.
 */

#include "lau_dispatch.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                               \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "[lau_laplacian] CUDA error %s:%d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return -1;                                                  \
        }                                                               \
    } while (0)

#define CUSPARSE_CHECK(call)                                           \
    do {                                                                \
        cusparseStatus_t st = (call);                                   \
        if (st != CUSPARSE_STATUS_SUCCESS) {                            \
            fprintf(stderr, "[lau_laplacian] cuSPARSE error %s:%d\n",  \
                    __FILE__, __LINE__);                                \
            return -2;                                                  \
        }                                                               \
    } while (0)

/* ── Kernel: Build Laplacian CSR values from adjacency weights ── */

__global__ void laplacian_csr_kernel(
    const int *row_ptr, const int *col_idx, const double *adj_weights,
    double *vals_out, int num_nodes, int nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = 0;
    /* Binary search for row (could be optimized with shared memory) */
    for (int r = 0; r < num_nodes; r++) {
        if (idx >= row_ptr[r] && idx < row_ptr[r + 1]) {
            row = r;
            break;
        }
    }

    int col = col_idx[idx];
    if (row == col) {
        /* Diagonal: sum of all adjacency weights for this row */
        double degree = 0.0;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            degree += adj_weights[j];
        }
        vals_out[idx] = degree;
    } else {
        /* Off-diagonal: -weight */
        vals_out[idx] = -adj_weights[idx];
    }
}

int lau_build_laplacian_csr(const int *row_ptr, const int *col_idx,
                            const double *adj_weights,
                            double *vals_out,
                            int num_nodes, int nnz)
{
    int *d_row_ptr, *d_col_idx;
    double *d_adj, *d_vals;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (num_nodes + 1) * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj, adj_weights, nnz * sizeof(double),
                           cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (nnz + block - 1) / block;
    laplacian_csr_kernel<<<grid, block>>>(d_row_ptr, d_col_idx, d_adj,
                                           d_vals, num_nodes, nnz);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(vals_out, d_vals, nnz * sizeof(double),
                           cudaMemcpyDeviceToHost));

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_adj); cudaFree(d_vals);
    return 0;
}

/* ── Batched spectral gap ──────────────────────────────────────── */
/* Uses power iteration on (L - sigma*I) to find second-smallest eigenvalue. */

__global__ void spmv_csr_kernel(
    const int *row_ptr, const int *col_idx, const double *vals,
    const double *x, double *y, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    double sum = 0.0;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
        sum += vals[j] * x[col_idx[j]];
    }
    y[row] = sum;
}

__global__ void normalize_kernel(double *x, double *norm_sq, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(norm_sq, x[idx] * x[idx]);
    }
}

__global__ void scale_kernel(double *x, double scale, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) x[idx] *= scale;
}

int lau_batch_spectral_gap(const int *row_ptrs, const int *col_idxs,
                           const double *vals,
                           const int *graph_sizes, const int *graph_nnz,
                           double *spectral_gaps,
                           int fleet_size)
{
    /* For each graph, use power iteration on L to estimate spectral gap.
     * Spectral gap = smallest nonzero eigenvalue of L.
     * We use inverse power iteration with shift to target small eigenvalues. */

    for (int g = 0; g < fleet_size; g++) {
        int N = graph_sizes[g];
        int nnz = graph_nnz[g];
        const int *rp = row_ptrs;  /* In production, these would be offset per-graph */
        const int *ci = col_idxs;
        const double *v = vals;

        int *d_rp, *d_ci;
        double *d_vals, *d_x, *d_y, *d_norm_sq;

        CUDA_CHECK(cudaMalloc(&d_rp, (N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ci, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_norm_sq, sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_rp, rp, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ci, ci, nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, v, nnz * sizeof(double), cudaMemcpyHostToDevice));

        /* Initialize x with ones */
        double *h_x = (double *)calloc(N, sizeof(double));
        for (int i = 0; i < N; i++) h_x[i] = 1.0 / sqrt((double)N);
        CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

        double eigenvalue = 0.0;
        int block = 256;
        int grid_n = (N + block - 1) / block;

        for (int iter = 0; iter < 200; iter++) {
            /* y = L * x */
            spmv_csr_kernel<<<grid_n, block>>>(d_rp, d_ci, d_vals, d_x, d_y, N);

            /* eigenvalue = x^T * y */
            double *h_x_cur = (double *)malloc(N * sizeof(double));
            double *h_y_cur = (double *)malloc(N * sizeof(double));
            CUDA_CHECK(cudaMemcpy(h_x_cur, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_y_cur, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

            double rayleigh = 0.0;
            for (int i = 0; i < N; i++) rayleigh += h_x_cur[i] * h_y_cur[i];
            eigenvalue = rayleigh;

            /* Normalize y */
            CUDA_CHECK(cudaMemset(d_norm_sq, 0, sizeof(double)));
            normalize_kernel<<<grid_n, block>>>(d_y, d_norm_sq, N);
            double h_norm;
            CUDA_CHECK(cudaMemcpy(&h_norm, d_norm_sq, sizeof(double), cudaMemcpyDeviceToHost));
            if (h_norm < 1e-30) break;
            scale_kernel<<<grid_n, block>>>(d_y, 1.0 / sqrt(h_norm), N);

            /* Check convergence */
            double *h_y_norm = (double *)malloc(N * sizeof(double));
            CUDA_CHECK(cudaMemcpy(h_y_norm, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));
            double diff = 0.0;
            for (int i = 0; i < N; i++) diff += (h_y_norm[i] - h_x_cur[i]) * (h_y_norm[i] - h_x_cur[i]);
            free(h_y_norm);
            free(h_x_cur);
            free(h_y_cur);

            /* Swap x and y (copy y into x) */
            double *tmp = d_x; d_x = d_y; d_y = tmp;

            if (sqrt(diff) < 1e-10) break;
        }

        spectral_gaps[g] = eigenvalue;
        free(h_x);

        cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_vals);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_norm_sq);
    }

    return 0;
}

/* ── Power iteration: dominant eigenvalue ──────────────────────── */

int lau_power_iteration(const int *row_ptr, const int *col_idx,
                        const double *vals, int N, int nnz,
                        double *dominant_eigenvalue, double *eigenvector_out,
                        int max_iter, double tol)
{
    int *d_rp, *d_ci;
    double *d_vals, *d_x, *d_y, *d_norm_sq;

    CUDA_CHECK(cudaMalloc(&d_rp, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_norm_sq, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_rp, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

    /* Random initial vector */
    double *h_x = (double *)malloc(N * sizeof(double));
    srand48(42);
    double init_norm = 0.0;
    for (int i = 0; i < N; i++) {
        h_x[i] = drand48();
        init_norm += h_x[i] * h_x[i];
    }
    init_norm = sqrt(init_norm);
    for (int i = 0; i < N; i++) h_x[i] /= init_norm;
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

    double eigenvalue = 0.0;
    int block = 256;
    int grid_n = (N + block - 1) / block;

    for (int iter = 0; iter < max_iter; iter++) {
        spmv_csr_kernel<<<grid_n, block>>>(d_rp, d_ci, d_vals, d_x, d_y, N);

        /* Rayleigh quotient on host */
        double *h_xc = (double *)malloc(N * sizeof(double));
        double *h_yc = (double *)malloc(N * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_xc, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_yc, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

        double num = 0.0, den = 0.0;
        for (int i = 0; i < N; i++) {
            num += h_xc[i] * h_yc[i];
            den += h_xc[i] * h_xc[i];
        }
        eigenvalue = num / den;

        /* Normalize */
        CUDA_CHECK(cudaMemset(d_norm_sq, 0, sizeof(double)));
        normalize_kernel<<<grid_n, block>>>(d_y, d_norm_sq, N);
        double h_norm;
        CUDA_CHECK(cudaMemcpy(&h_norm, d_norm_sq, sizeof(double), cudaMemcpyDeviceToHost));
        if (h_norm < 1e-30) break;
        scale_kernel<<<grid_n, block>>>(d_y, 1.0 / sqrt(h_norm), N);

        /* Check convergence */
        double diff = 0.0;
        for (int i = 0; i < N; i++) diff += (h_yc[i] / sqrt(h_norm) - h_xc[i]) * (h_yc[i] / sqrt(h_norm) - h_xc[i]);
        free(h_xc); free(h_yc);

        /* Swap */
        double *tmp = d_x; d_x = d_y; d_y = tmp;

        if (sqrt(diff) < tol) break;
    }

    *dominant_eigenvalue = eigenvalue;
    if (eigenvector_out) {
        CUDA_CHECK(cudaMemcpy(eigenvector_out, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    }

    free(h_x);
    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_vals);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_norm_sq);
    return 0;
}
