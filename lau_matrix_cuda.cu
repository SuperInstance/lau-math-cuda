/*
 * lau_matrix_cuda.cu — CuBLAS / cuSOLVER matrix operations
 *
 * Implements: matmul, batched inverse, batched eigendecomposition
 * for matrices 16x16 .. 4096x4096.
 */

#include "lau_dispatch.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CUDA_CHECK(call)                                               \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "[lau_matrix] CUDA error %s:%d: %s\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return -1;                                                  \
        }                                                               \
    } while (0)

#define CUBLAS_CHECK(call)                                             \
    do {                                                                \
        cublasStatus_t st = (call);                                     \
        if (st != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "[lau_matrix] cuBLAS error %s:%d: %d\n",   \
                    __FILE__, __LINE__, st);                            \
            return -2;                                                  \
        }                                                               \
    } while (0)

#define CUSOLVER_CHECK(call)                                           \
    do {                                                                \
        cusolverStatus_t st = (call);                                   \
        if (st != CUSOLVER_STATUS_SUCCESS) {                            \
            fprintf(stderr, "[lau_matrix] cuSOLVER error %s:%d: %d\n", \
                    __FILE__, __LINE__, st);                            \
            return -3;                                                  \
        }                                                               \
    } while (0)

/* ── Matrix multiply C = alpha * A * B + beta * C ────────────── */

int lau_matmul(const double *h_A, const double *h_B, double *h_C,
               int M, int K, int N, int batch_size)
{
    if (batch_size < 1) return -1;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    size_t a_bytes = (size_t)batch_size * M * K * sizeof(double);
    size_t b_bytes = (size_t)batch_size * K * N * sizeof(double);
    size_t c_bytes = (size_t)batch_size * M * N * sizeof(double);

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));

    const double alpha = 1.0, beta = 0.0;

    if (batch_size == 1) {
        /* Single multiply: C = A * B */
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    } else {
        /* Batched: strided DGEMM */
        long long int strideA = (long long int)(M * K);
        long long int strideB = (long long int)(K * N);
        long long int strideC = (long long int)(M * N);
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, N, strideB,
            d_A, K, strideA,
            &beta,
            d_C, N, strideC,
            batch_size));
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}

/* ── Batched matrix inverse via cuSOLVER getrf + getri ────────── */

int lau_batch_inverse(const double *h_A, double *h_Ainv,
                      int N, int batch_size)
{
    if (N < 1 || batch_size < 1) return -1;

    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    size_t mat_bytes = (size_t)N * N * sizeof(double);
    size_t total_bytes = (size_t)batch_size * mat_bytes;
    int *d_pivot, *d_info;
    double *d_A, *d_work;
    int lwork = 0;

    CUDA_CHECK(cudaMalloc(&d_A, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, total_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_pivot, (size_t)batch_size * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, batch_size * sizeof(int)));

    /* Query workspace for batched getrf */
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solver, N, N, d_A, N, &lwork));
    CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(double)));

    for (int b = 0; b < batch_size; b++) {
        double *Ab = d_A + (size_t)b * N * N;
        int *pb = d_pivot + (size_t)b * N;
        int *ib = d_info + b;

        CUSOLVER_CHECK(cusolverDnDgetrf(solver, N, N, Ab, N, d_work, pb, ib));

        /* Check singular */
        int h_info;
        CUDA_CHECK(cudaMemcpy(&h_info, ib, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            fprintf(stderr, "[lau_matrix] Matrix %d is singular (info=%d)\n", b, h_info);
            cudaFree(d_A); cudaFree(d_pivot); cudaFree(d_info); cudaFree(d_work);
            cusolverDnDestroy(solver);
            return -4;
        }
    }

    /* Now compute inverse via getri for each batch */
    for (int b = 0; b < batch_size; b++) {
        double *Ab = d_A + (size_t)b * N * N;
        int *pb = d_pivot + (size_t)b * N;
        int *ib = d_info + b;

        CUSOLVER_CHECK(cusolverDnDgetri(solver, N, Ab, N, pb, d_work, lwork, ib));
    }

    CUDA_CHECK(cudaMemcpy(h_Ainv, d_A, total_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_pivot); cudaFree(d_info); cudaFree(d_work);
    cusolverDnDestroy(solver);
    return 0;
}

/* ── Batched eigendecomposition via cuSOLVER syevd ────────────── */

int lau_batch_eigendecompose(const double *h_A, double *h_eigenvalues,
                             double *h_eigenvectors, int N, int batch_size)
{
    if (N < 1 || batch_size < 1) return -1;

    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    size_t mat_bytes = (size_t)N * N * sizeof(double);
    size_t total_bytes = (size_t)batch_size * mat_bytes;

    double *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, total_bytes, cudaMemcpyHostToDevice));

    double *d_W; /* eigenvalues */
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)batch_size * N * sizeof(double)));

    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_info, batch_size * sizeof(int)));

    for (int b = 0; b < batch_size; b++) {
        double *Ab = d_A + (size_t)b * N * N;
        double *Wb = d_W + (size_t)b * N;
        int *ib = d_info + b;

        /* Query workspace */
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
            solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            N, Ab, N, Wb, &lwork));

        double *d_work;
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(double)));

        CUSOLVER_CHECK(cusolverDnDsyevd(
            solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            N, Ab, N, Wb, d_work, lwork, ib));

        cudaFree(d_work);
    }

    /* eigenvectors are in d_A (overwritten) */
    CUDA_CHECK(cudaMemcpy(h_eigenvectors, d_A, total_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_eigenvalues, d_W,
                          (size_t)batch_size * N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_W); cudaFree(d_info);
    cusolverDnDestroy(solver);
    return 0;
}
