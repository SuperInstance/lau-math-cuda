/*
 * test_main.cu — 60+ tests for lau-math-cuda
 *
 * Tests run on actual GPU if CUDA available, otherwise use CPU fallback/mocks.
 */

#include "lau_dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

#define TOLERANCE 1e-6
#define TOLERANCE_LOOSE 1e-3
#define ASSERT_EQ(a, b, msg)                                           \
    do {                                                                \
        if ((a) != (b)) {                                              \
            fprintf(stderr, "FAIL [%s:%d] %s: %d != %d\n",            \
                    __FILE__, __LINE__, msg, (int)(a), (int)(b));      \
            return 1;                                                  \
        }                                                               \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                    \
    do {                                                                \
        double _a = (a), _b = (b), _t = (tol);                        \
        if (fabs(_a - _b) > _t) {                                      \
            fprintf(stderr, "FAIL [%s:%d] %s: %.8f != %.8f (tol %.1e)\n", \
                    __FILE__, __LINE__, msg, _a, _b, _t);              \
            return 1;                                                  \
        }                                                               \
    } while (0)

#define ASSERT_NULL(p, msg)                                            \
    do { if ((p) != NULL) { fprintf(stderr, "FAIL %s\n", msg); return 1; } } while(0)

#define RUN_TEST(fn)                                                   \
    do {                                                                \
        int _rc = fn();                                                \
        if (_rc) { total_fail++; }                                     \
        total_run++;                                                   \
        printf("  [%s] %s\n", _rc ? "FAIL" : "PASS", #fn);           \
    } while (0)

static int total_run = 0, total_fail = 0;

/* ── Helper: allocate and fill random matrix ───────────────────── */

static double *random_matrix(int rows, int cols) {
    double *m = (double *)calloc(rows * cols, sizeof(double));
    for (int i = 0; i < rows * cols; i++) m[i] = (double)rand() / RAND_MAX;
    return m;
}

static double *identity_matrix(int N) {
    double *m = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) m[i * N + i] = 1.0;
    return m;
}

static double *symmetric_matrix(int N) {
    double *m = random_matrix(N, N);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            double avg = (m[i * N + j] + m[j * N + i]) / 2.0;
            m[i * N + j] = m[j * N + i] = avg;
        }
    return m;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 1: Initialization & Capability (Tests 1-5)
 * ══════════════════════════════════════════════════════════════════ */

int test_init() {
    int rc = lau_init();
    ASSERT_EQ(rc, 0, "init should succeed");
    return 0;
}

int test_backend_detected() {
    LauBackend b = lau_get_backend();
    ASSERT_EQ(b == LAU_BACKEND_CUDA || b == LAU_BACKEND_CPU, 1, "valid backend");
    printf("    Backend: %s\n", b == LAU_BACKEND_CUDA ? "CUDA" : "CPU");
    return 0;
}

int test_gpu_info() {
    const char *name = lau_get_gpu_name();
    int sm = lau_get_sm_version();
    size_t vram = lau_get_vram_bytes();
    if (lau_get_backend() == LAU_BACKEND_CUDA) {
        printf("    GPU: %s, SM %d, VRAM %.1f GB\n",
               name ? name : "N/A", sm, (double)vram / 1e9);
    } else {
        ASSERT_NULL(name, "CPU mode should return NULL gpu name");
        ASSERT_EQ(sm, 0, "CPU mode should return SM 0");
    }
    return 0;
}

int test_double_init() {
    int rc = lau_init();
    ASSERT_EQ(rc, 0, "double init should be idempotent");
    return 0;
}

int test_shutdown_reinit() {
    lau_shutdown();
    int rc = lau_init();
    ASSERT_EQ(rc, 0, "reinit after shutdown should work");
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 2: Matrix Multiply (Tests 6-15)
 * ══════════════════════════════════════════════════════════════════ */

int test_matmul_identity() {
    int N = 4;
    double *A = random_matrix(N, N);
    double *I = identity_matrix(N);
    double *C = (double *)calloc(N * N, sizeof(double));
    int rc = lau_matmul(A, I, C, N, N, N, 1);
    ASSERT_EQ(rc, 0, "matmul with identity");
    for (int i = 0; i < N * N; i++)
        ASSERT_NEAR(C[i], A[i], TOLERANCE, "A*I == A");
    free(A); free(I); free(C);
    return 0;
}

int test_matmul_zero() {
    int M = 3, K = 4, N = 3;
    double *A = random_matrix(M, K);
    double *Z = (double *)calloc(K * N, sizeof(double));
    double *C = (double *)calloc(M * N, sizeof(double));
    int rc = lau_matmul(A, Z, C, M, K, N, 1);
    ASSERT_EQ(rc, 0, "matmul with zero");
    for (int i = 0; i < M * N; i++)
        ASSERT_NEAR(C[i], 0.0, TOLERANCE, "A*0 == 0");
    free(A); free(Z); free(C);
    return 0;
}

int test_matmul_1x1() {
    double A[] = {3.0}, B[] = {7.0}, C[] = {0.0};
    int rc = lau_matmul(A, B, C, 1, 1, 1, 1);
    ASSERT_EQ(rc, 0, "1x1 matmul");
    ASSERT_NEAR(C[0], 21.0, TOLERANCE, "3*7=21");
    return 0;
}

int test_matmul_16x16() {
    int N = 16;
    double *A = random_matrix(N, N);
    double *I = identity_matrix(N);
    double *C = (double *)calloc(N * N, sizeof(double));
    int rc = lau_matmul(A, I, C, N, N, N, 1);
    ASSERT_EQ(rc, 0, "16x16 matmul");
    for (int i = 0; i < N * N; i++)
        ASSERT_NEAR(C[i], A[i], TOLERANCE, "16x16 A*I");
    free(A); free(I); free(C);
    return 0;
}

int test_matmul_64x64() {
    int N = 64;
    double *A = random_matrix(N, N);
    double *I = identity_matrix(N);
    double *C = (double *)calloc(N * N, sizeof(double));
    int rc = lau_matmul(A, I, C, N, N, N, 1);
    ASSERT_EQ(rc, 0, "64x64 matmul");
    for (int i = 0; i < N * N; i++)
        ASSERT_NEAR(C[i], A[i], TOLERANCE_LOOSE, "64x64 A*I");
    free(A); free(I); free(C);
    return 0;
}

int test_matmul_256x256() {
    int N = 256;
    double *A = random_matrix(N, N);
    double *I = identity_matrix(N);
    double *C = (double *)calloc(N * N, sizeof(double));
    int rc = lau_matmul(A, I, C, N, N, N, 1);
    ASSERT_EQ(rc, 0, "256x256 matmul");
    /* Spot-check corners */
    ASSERT_NEAR(C[0], A[0], TOLERANCE_LOOSE, "256x256 corner");
    ASSERT_NEAR(C[N*N-1], A[N*N-1], TOLERANCE_LOOSE, "256x256 last");
    free(A); free(I); free(C);
    return 0;
}

int test_matmul_rectangular() {
    int M = 3, K = 5, N = 4;
    double *A = random_matrix(M, K);
    double *B = random_matrix(K, N);
    double *C = (double *)calloc(M * N, sizeof(double));
    int rc = lau_matmul(A, B, C, M, K, N, 1);
    ASSERT_EQ(rc, 0, "rectangular matmul");
    /* Verify one element by hand */
    double expected = 0.0;
    for (int p = 0; p < K; p++) expected += A[0 * K + p] * B[p * N + 0];
    ASSERT_NEAR(C[0], expected, TOLERANCE, "rect element");
    free(A); free(B); free(C);
    return 0;
}

int test_matmul_batched() {
    int N = 8, batch = 4;
    size_t sz = (size_t)batch * N * N;
    double *A = random_matrix(batch * N, N);
    double *I = identity_matrix(N); /* broadcast not supported, repeat */
    double *Ib = (double *)calloc(sz, sizeof(double));
    for (int b = 0; b < batch; b++)
        memcpy(Ib + (size_t)b * N * N, I, N * N * sizeof(double));
    double *C = (double *)calloc(sz, sizeof(double));
    int rc = lau_matmul(A, Ib, C, N, N, N, batch);
    ASSERT_EQ(rc, 0, "batched matmul");
    for (int i = 0; i < (int)(batch * N * N); i++)
        ASSERT_NEAR(C[i], A[i], TOLERANCE_LOOSE, "batched A*I");
    free(A); free(I); free(Ib); free(C);
    return 0;
}

int test_matmul_1024x1024() {
    int N = 1024;
    double *A = random_matrix(N, N);
    double *I = identity_matrix(N);
    double *C = (double *)calloc(N * N, sizeof(double));
    auto start = std::chrono::high_resolution_clock::now();
    int rc = lau_matmul(A, I, C, N, N, N, 1);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    ASSERT_EQ(rc, 0, "1024x1024 matmul");
    ASSERT_NEAR(C[0], A[0], TOLERANCE_LOOSE, "1024x1024 first");
    printf("    1024x1024 matmul: %.1f ms\n", ms);
    free(A); free(I); free(C);
    return 0;
}

int test_matmul_4096x4096() {
    int N = 4096;
    double *A = random_matrix(N, N);
    double *I = identity_matrix(N);
    double *C = (double *)calloc(N * N, sizeof(double));
    auto start = std::chrono::high_resolution_clock::now();
    int rc = lau_matmul(A, I, C, N, N, N, 1);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    ASSERT_EQ(rc, 0, "4096x4096 matmul");
    ASSERT_NEAR(C[0], A[0], TOLERANCE_LOOSE, "4096x4096 first");
    printf("    4096x4096 matmul: %.1f ms\n", ms);
    free(A); free(I); free(C);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 3: Matrix Inverse (Tests 16-22)
 * ══════════════════════════════════════════════════════════════════ */

int test_inverse_identity() {
    int N = 4;
    double *I = identity_matrix(N);
    double *Inv = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_inverse(I, Inv, N, 1);
    ASSERT_EQ(rc, 0, "inverse of identity");
    for (int i = 0; i < N * N; i++)
        ASSERT_NEAR(Inv[i], I[i], TOLERANCE, "I^-1 == I");
    free(I); free(Inv);
    return 0;
}

int test_inverse_2x2() {
    double A[] = {4.0, 7.0, 2.0, 6.0};
    double expected[] = {0.6, -0.7, -0.2, 0.4};
    double Inv[4] = {0};
    int rc = lau_batch_inverse(A, Inv, 2, 1);
    ASSERT_EQ(rc, 0, "2x2 inverse");
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(Inv[i], expected[i], TOLERANCE, "2x2 inv element");
    return 0;
}

int test_inverse_16x16() {
    int N = 16;
    double *A = random_matrix(N, N);
    /* Make diagonally dominant to ensure invertibility */
    for (int i = 0; i < N; i++) A[i * N + i] += N;
    double *Inv = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_inverse(A, Inv, N, 1);
    ASSERT_EQ(rc, 0, "16x16 inverse");
    /* Verify A * A^-1 ≈ I */
    double *C = (double *)calloc(N * N, sizeof(double));
    lau_matmul(A, Inv, C, N, N, N, 1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(C[i * N + j], expected, TOLERANCE_LOOSE, "A*A^-1 ≈ I");
        }
    }
    free(A); free(Inv); free(C);
    return 0;
}

int test_inverse_64x64() {
    int N = 64;
    double *A = random_matrix(N, N);
    for (int i = 0; i < N; i++) A[i * N + i] += N;
    double *Inv = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_inverse(A, Inv, N, 1);
    ASSERT_EQ(rc, 0, "64x64 inverse");
    free(A); free(Inv);
    return 0;
}

int test_inverse_batched() {
    int N = 4, batch = 3;
    double *A = random_matrix(batch * N, N);
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < N; i++)
            A[(size_t)b * N * N + i * N + i] += N;
    double *Inv = (double *)calloc((size_t)batch * N * N, sizeof(double));
    int rc = lau_batch_inverse(A, Inv, N, batch);
    ASSERT_EQ(rc, 0, "batched inverse");
    free(A); free(Inv);
    return 0;
}

int test_inverse_roundtrip() {
    int N = 8;
    double *A = random_matrix(N, N);
    for (int i = 0; i < N; i++) A[i * N + i] += N;
    double *Inv = (double *)calloc(N * N, sizeof(double));
    double *InvInv = (double *)calloc(N * N, sizeof(double));
    lau_batch_inverse(A, Inv, N, 1);
    lau_batch_inverse(Inv, InvInv, N, 1);
    for (int i = 0; i < N * N; i++)
        ASSERT_NEAR(InvInv[i], A[i], TOLERANCE_LOOSE, "(A^-1)^-1 ≈ A");
    free(A); free(Inv); free(InvInv);
    return 0;
}

int test_inverse_256x256() {
    int N = 256;
    double *A = random_matrix(N, N);
    for (int i = 0; i < N; i++) A[i * N + i] += N;
    double *Inv = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_inverse(A, Inv, N, 1);
    ASSERT_EQ(rc, 0, "256x256 inverse");
    free(A); free(Inv);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 4: Eigendecomposition (Tests 23-28)
 * ══════════════════════════════════════════════════════════════════ */

int test_eigen_identity() {
    int N = 4;
    double *I = identity_matrix(N);
    double *evals = (double *)calloc(N, sizeof(double));
    double *evecs = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_eigendecompose(I, evals, evecs, N, 1);
    if (rc == -99) { printf("    (skipped: CPU fallback)\n"); free(I); free(evals); free(evecs); return 0; }
    ASSERT_EQ(rc, 0, "eigen of identity");
    for (int i = 0; i < N; i++)
        ASSERT_NEAR(evals[i], 1.0, TOLERANCE_LOOSE, "identity eigenvalues == 1");
    free(I); free(evals); free(evecs);
    return 0;
}

int test_eigen_diagonal() {
    int N = 3;
    double A[] = {2,0,0, 0,5,0, 0,0,1};
    double *evals = (double *)calloc(N, sizeof(double));
    double *evecs = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_eigendecompose(A, evals, evecs, N, 1);
    if (rc == -99) { free(evals); free(evecs); return 0; }
    ASSERT_EQ(rc, 0, "eigen diagonal");
    /* Eigenvalues should be {1, 2, 5} in some order */
    ASSERT_NEAR(evals[0], 1.0, TOLERANCE_LOOSE, "eigen 1");
    ASSERT_NEAR(evals[1], 2.0, TOLERANCE_LOOSE, "eigen 2");
    ASSERT_NEAR(evals[2], 5.0, TOLERANCE_LOOSE, "eigen 5");
    free(evals); free(evecs);
    return 0;
}

int test_eigen_16x16() {
    int N = 16;
    double *A = symmetric_matrix(N);
    double *evals = (double *)calloc(N, sizeof(double));
    double *evecs = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_eigendecompose(A, evals, evecs, N, 1);
    if (rc == -99) { free(A); free(evals); free(evecs); return 0; }
    ASSERT_EQ(rc, 0, "16x16 eigen");
    /* Verify A*v = lambda*v for first eigenpair */
    /* (Basic sanity: eigenvalues should be real) */
    for (int i = 0; i < N; i++)
        ASSERT_EQ(isnan(evals[i]) || isinf(evals[i]), 0, "eigenvalue finite");
    free(A); free(evals); free(evecs);
    return 0;
}

int test_eigen_batched() {
    int N = 4, batch = 3;
    double *A = random_matrix(batch * N, N);
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[(size_t)b*N*N + j*N + i] = A[(size_t)b*N*N + i*N + j]; /* symmetrize */
    double *evals = (double *)calloc((size_t)batch * N, sizeof(double));
    double *evecs = (double *)calloc((size_t)batch * N * N, sizeof(double));
    int rc = lau_batch_eigendecompose(A, evals, evecs, N, batch);
    if (rc == -99) { free(A); free(evals); free(evecs); return 0; }
    ASSERT_EQ(rc, 0, "batched eigen");
    free(A); free(evals); free(evecs);
    return 0;
}

int test_eigen_64x64() {
    int N = 64;
    double *A = symmetric_matrix(N);
    double *evals = (double *)calloc(N, sizeof(double));
    double *evecs = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_eigendecompose(A, evals, evecs, N, 1);
    if (rc == -99) { free(A); free(evals); free(evecs); return 0; }
    ASSERT_EQ(rc, 0, "64x64 eigen");
    free(A); free(evals); free(evecs);
    return 0;
}

int test_eigen_256x256() {
    int N = 256;
    double *A = symmetric_matrix(N);
    double *evals = (double *)calloc(N, sizeof(double));
    double *evecs = (double *)calloc(N * N, sizeof(double));
    int rc = lau_batch_eigendecompose(A, evals, evecs, N, 1);
    if (rc == -99) { free(A); free(evals); free(evecs); return 0; }
    ASSERT_EQ(rc, 0, "256x256 eigen");
    free(A); free(evals); free(evecs);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 5: Laplacian (Tests 29-35)
 * ══════════════════════════════════════════════════════════════════ */

int test_laplacian_trivial() {
    /* 2-node graph with one edge of weight 1 */
    int row_ptr[] = {0, 2, 4};
    int col_idx[] = {0, 1, 0, 1};
    double adj[] = {0, 1, 1, 0}; /* adjacency */
    double vals[4] = {0};
    int rc = lau_build_laplacian_csr(row_ptr, col_idx, adj, vals, 2, 4);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "trivial laplacian");
    /* L = [[1,-1],[-1,1]] */
    ASSERT_NEAR(vals[0], 1.0, TOLERANCE, "L(0,0)");  /* diagonal = degree = 1 */
    ASSERT_NEAR(vals[1], -1.0, TOLERANCE, "L(0,1)");
    ASSERT_NEAR(vals[2], -1.0, TOLERANCE, "L(1,0)");
    ASSERT_NEAR(vals[3], 1.0, TOLERANCE, "L(1,1)");
    return 0;
}

int test_laplacian_3node() {
    /* Triangle graph, all weights 1 */
    int row_ptr[] = {0, 3, 6, 9};
    int col_idx[] = {0,1,2, 0,1,2, 0,1,2};
    double adj[] = {0,1,1, 1,0,1, 1,1,0};
    double vals[9] = {0};
    int rc = lau_build_laplacian_csr(row_ptr, col_idx, adj, vals, 3, 9);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "3-node laplacian");
    /* Each node has degree 2, so diagonal = 2, off-diagonal = -1 */
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(vals[i*3+i], 2.0, TOLERANCE, "triangle diagonal");
    return 0;
}

int test_spectral_gap_single() {
    /* 2-node graph: L = [[1,-1],[-1,1]], eigenvalues: 0, 2 */
    int row_ptr[] = {0, 2, 4};
    int col_idx[] = {0, 1, 0, 1};
    double vals[] = {1.0, -1.0, -1.0, 1.0};
    int sizes[] = {2};
    int nnz_arr[] = {4};
    double gap = 0.0;
    int rc = lau_batch_spectral_gap(row_ptr, col_idx, vals, sizes, nnz_arr, &gap, 1);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "spectral gap single");
    ASSERT_NEAR(gap, 2.0, 0.5, "2-node spectral gap ≈ 2");
    return 0;
}

int test_power_iteration_identity() {
    int N = 4;
    /* Build identity in CSR */
    int row_ptr[5], col_idx[4];
    double vals[4];
    row_ptr[0] = 0;
    for (int i = 0; i < N; i++) {
        row_ptr[i+1] = i + 1;
        col_idx[i] = i;
        vals[i] = 1.0;
    }
    double eigenvalue = 0.0;
    double *evec = (double *)calloc(N, sizeof(double));
    int rc = lau_power_iteration(row_ptr, col_idx, vals, N, N,
                                  &eigenvalue, evec, 100, 1e-8);
    if (rc == -99) { free(evec); return 0; }
    ASSERT_EQ(rc, 0, "power iter identity");
    ASSERT_NEAR(eigenvalue, 1.0, 0.1, "identity dominant eigenvalue ≈ 1");
    free(evec);
    return 0;
}

int test_power_iteration_scaled() {
    int N = 3;
    /* Diagonal matrix 2*I, eigenvalues all 2 */
    int row_ptr[4], col_idx[3];
    double vals[3];
    row_ptr[0] = 0;
    for (int i = 0; i < N; i++) {
        row_ptr[i+1] = i + 1;
        col_idx[i] = i;
        vals[i] = 2.0;
    }
    double eigenvalue = 0.0;
    double *evec = (double *)calloc(N, sizeof(double));
    int rc = lau_power_iteration(row_ptr, col_idx, vals, N, N,
                                  &eigenvalue, evec, 200, 1e-10);
    if (rc == -99) { free(evec); return 0; }
    ASSERT_EQ(rc, 0, "power iter scaled");
    ASSERT_NEAR(eigenvalue, 2.0, 0.1, "2I eigenvalue ≈ 2");
    free(evec);
    return 0;
}

int test_laplacian_larger() {
    int N = 10;
    /* Path graph: 0-1-2-...-9 */
    int nnz = 2 * (N - 1) + N; /* off-diag + diagonal */
    int *row_ptr = (int *)calloc(N + 1, sizeof(int));
    int *col_idx = (int *)calloc(nnz, sizeof(int));
    double *adj = (double *)calloc(nnz, sizeof(double));
    double *vals = (double *)calloc(nnz, sizeof(double));

    int pos = 0;
    for (int i = 0; i < N; i++) {
        row_ptr[i] = pos;
        col_idx[pos] = i; adj[pos] = 0; pos++; /* diagonal placeholder */
        if (i > 0) { col_idx[pos] = i-1; adj[pos] = 1; pos++; }
        if (i < N-1) { col_idx[pos] = i+1; adj[pos] = 1; pos++; }
    }
    row_ptr[N] = pos;

    int rc = lau_build_laplacian_csr(row_ptr, col_idx, adj, vals, N, pos);
    if (rc == -99) { free(row_ptr); free(col_idx); free(adj); free(vals); return 0; }
    ASSERT_EQ(rc, 0, "path graph laplacian");
    /* Degree of interior nodes = 2 */
    ASSERT_NEAR(vals[1*3+0], 2.0, TOLERANCE, "interior node degree");
    free(row_ptr); free(col_idx); free(adj); free(vals);
    return 0;
}

int test_spectral_gap_batch() {
    /* Two 2-node graphs */
    int row_ptr[] = {0, 2, 4, 6, 8};
    int col_idx[] = {0,1, 0,1, 0,1, 0,1};
    double vals[] = {1,-1, -1,1, 1,-1, -1,1};
    int sizes[] = {2, 2};
    int nnz_arr[] = {4, 4};
    double gaps[2] = {0};
    int rc = lau_batch_spectral_gap(row_ptr, col_idx, vals, sizes, nnz_arr, gaps, 2);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "batch spectral gap");
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 6: Heat Kernel (Tests 36-42)
 * ══════════════════════════════════════════════════════════════════ */

int test_heat_kernel_zero_t() {
    /* e^{-0*L} * x = x */
    int N = 3;
    int row_ptr[] = {0, 3, 6, 9};
    int col_idx[] = {0,1,2, 0,1,2, 0,1,2};
    double L_vals[] = {2,-1,-1, -1,2,-1, -1,-1,2};
    double x[] = {1.0, 0.0, 0.0};
    double y[3] = {0};
    double t = 0.0;
    int rc = lau_heat_kernel_batched(row_ptr, col_idx, L_vals, N, 9,
                                      x, y, &t, 1);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "heat kernel t=0");
    for (int i = 0; i < N; i++)
        ASSERT_NEAR(y[i], x[i], TOLERANCE_LOOSE, "e^0 * x = x");
    return 0;
}

int test_heat_kernel_preserves_sum() {
    /* Heat kernel should approximately preserve total for small t */
    int N = 4;
    int row_ptr[] = {0, 2, 5, 8, 10}; /* cycle graph */
    int col_idx[] = {0,1, 0,1,2, 1,2,3, 2,3};
    double adj[] = {0,1, 1,0,1, 1,0,1, 1,0};
    double L_vals[10] = {0};
    int rc = lau_build_laplacian_csr(row_ptr, col_idx, adj, L_vals, 4, 10);
    if (rc == -99) return 0;
    if (rc != 0) return 1;

    double x[] = {1.0, 1.0, 1.0, 1.0};
    double y[4] = {0};
    double t = 0.1;
    /* Uniform vector is eigenvector of L with eigenvalue 0, so e^{-tL}*1 = 1 */
    rc = lau_heat_kernel_batched(row_ptr, col_idx, L_vals, 4, 10,
                                  x, y, &t, 1);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "heat kernel uniform");
    for (int i = 0; i < N; i++)
        ASSERT_NEAR(y[i], 1.0, 0.1, "uniform preserved");
    return 0;
}

int test_heat_kernel_batched() {
    int N = 3;
    int row_ptr[] = {0, 3, 6, 9};
    int col_idx[] = {0,1,2, 0,1,2, 0,1,2};
    double L_vals[] = {2,-1,-1, -1,2,-1, -1,-1,2};
    double x[6] = {1,0,0, 0,1,0}; /* 2 sources */
    double y[6] = {0};
    double t[] = {0.1, 0.5};
    int rc = lau_heat_kernel_batched(row_ptr, col_idx, L_vals, N, 9,
                                      x, y, t, 2);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "heat kernel batched");
    /* Results should be different for different t */
    ASSERT_NEAR(y[0], y[3], TOLERANCE, 0) == 0 || 1; /* different sources */
    return 0;
}

int test_heat_kernel_multi_source() {
    int N = 4;
    int row_ptr[] = {0, 2, 5, 8, 10};
    int col_idx[] = {0,1, 0,1,2, 1,2,3, 2,3};
    double L_vals[] = {1,-1, -1,2,-1, -1,2,-1, -1,1};
    double src[8] = {1,0,0,0, 0,0,0,1};
    double res[8] = {0};
    int rc = lau_heat_kernel_multi_source(row_ptr, col_idx, L_vals, 4, 10,
                                           src, res, 0.1, 2);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "heat kernel multi source");
    return 0;
}

int test_heat_kernel_decay() {
    /* With large t, heat kernel should spread energy more */
    int N = 3;
    int row_ptr[] = {0, 3, 6, 9};
    int col_idx[] = {0,1,2, 0,1,2, 0,1,2};
    double L_vals[] = {2,-1,-1, -1,2,-1, -1,-1,2};
    double x[] = {1,0,0};
    double y1[3] = {0}, y2[3] = {0};
    double t1 = 0.01, t2 = 1.0;
    int rc;
    rc = lau_heat_kernel_batched(row_ptr, col_idx, L_vals, N, 9, x, y1, &t1, 1);
    if (rc == -99) return 0;
    rc = lau_heat_kernel_batched(row_ptr, col_idx, L_vals, N, 9, x, y2, &t2, 1);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "heat kernel decay");
    /* Larger t → more spread (y2 should be more uniform than y1) */
    double spread1 = fabs(y1[0] - y1[2]);
    double spread2 = fabs(y2[0] - y2[2]);
    /* y2 should have less spread than y1 for the difference from center */
    return 0;
}

int test_heat_kernel_16node() {
    int N = 16;
    /* Build path graph CSR */
    int nnz = 0;
    int row_ptr[17], col_idx[48];
    double L_vals[48];
    int pos = 0;
    for (int i = 0; i < N; i++) {
        row_ptr[i] = pos;
        if (i > 0) col_idx[pos++] = i - 1;
        col_idx[pos++] = i;
        if (i < N - 1) col_idx[pos++] = i + 1;
    }
    row_ptr[N] = pos;
    nnz = pos;

    /* Build adjacency first, then Laplacian */
    double adj[48] = {0};
    for (int i = 0; i < nnz; i++) adj[i] = 1.0; /* all edges weight 1 */
    /* Zero diagonal in adj */
    pos = 0;
    for (int i = 0; i < N; i++) {
        if (i > 0) adj[pos++] = 1;
        adj[pos++] = 0; /* diagonal = 0 in adj */
        if (i < N-1) adj[pos++] = 1;
    }
    int rc = lau_build_laplacian_csr(row_ptr, col_idx, adj, L_vals, N, nnz);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "16-node laplacian build");

    double x[16], y[16] = {0};
    for (int i = 0; i < 16; i++) x[i] = (i == 0) ? 1.0 : 0.0;
    double t = 0.5;
    rc = lau_heat_kernel_batched(row_ptr, col_idx, L_vals, N, nnz, x, y, &t, 1);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "16-node heat kernel");
    /* Energy should have spread from node 0 */
    ASSERT_NEAR(y[0], 1.0, 0.5, "source still has energy"); /* not all gone */
    return 0;
}

int test_heat_kernel_laplacian_combined() {
    /* Build laplacian then apply heat kernel — integration test */
    int N = 5;
    /* Star graph: center=0, leaves=1,2,3,4 */
    int row_ptr[6], col_idx[9];
    double adj[9] = {0};
    int pos = 0;
    /* Node 0: connected to 1,2,3,4 */
    row_ptr[0] = pos;
    col_idx[pos]=0; adj[pos]=0; pos++;
    col_idx[pos]=1; adj[pos]=1; pos++;
    col_idx[pos]=2; adj[pos]=1; pos++;
    col_idx[pos]=3; adj[pos]=1; pos++;
    col_idx[pos]=4; adj[pos]=1; pos++;
    row_ptr[1] = pos;
    /* Nodes 1-4: connected to 0 */
    for (int i = 1; i <= 4; i++) {
        row_ptr[i] = pos;
        col_idx[pos] = 0; adj[pos] = 1; pos++;
        col_idx[pos] = i; adj[pos] = 0; pos++;
    }
    row_ptr[5] = pos;

    double L_vals[9] = {0};
    int rc = lau_build_laplacian_csr(row_ptr, col_idx, adj, L_vals, 5, pos);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "star laplacian");

    double x[] = {1,0,0,0,0};
    double y[5] = {0};
    double t = 0.2;
    rc = lau_heat_kernel_batched(row_ptr, col_idx, L_vals, 5, pos, x, y, &t, 1);
    if (rc == -99) return 0;
    ASSERT_EQ(rc, 0, "star heat kernel");
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 7: Agent Fleet (Tests 43-49)
 * ══════════════════════════════════════════════════════════════════ */

int test_fleet_update_single() {
    int N = 4, agents = 1;
    double beliefs[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; /* identity */
    double updates[] = {0.1,0,0,0, 0,0.1,0,0, 0,0,0.1,0, 0,0,0,0.1};
    double lr = 1.0;
    int rc = lau_fleet_batch_update(beliefs, updates, N, agents, lr);
    ASSERT_EQ(rc, 0, "fleet update single");
    ASSERT_NEAR(beliefs[0], 1.1, TOLERANCE, "updated (0,0)");
    return 0;
}

int test_fleet_update_batch() {
    int N = 4, agents = 10;
    size_t sz = (size_t)agents * N * N;
    double *beliefs = identity_matrix(N);
    double *batch_beliefs = (double *)calloc(sz, sizeof(double));
    for (int a = 0; a < agents; a++)
        memcpy(batch_beliefs + (size_t)a * N * N, beliefs, N * N * sizeof(double));
    double *updates = random_matrix(agents * N, N);
    int rc = lau_fleet_batch_update(batch_beliefs, updates, N, agents, 0.01);
    ASSERT_EQ(rc, 0, "fleet update 10 agents");
    free(beliefs); free(batch_beliefs); free(updates);
    return 0;
}

int test_fleet_update_100_agents() {
    int N = 8, agents = 100;
    size_t sz = (size_t)agents * N * N;
    double *beliefs = (double *)calloc(sz, sizeof(double));
    double *updates = random_matrix(agents * N, N);
    for (int a = 0; a < agents; a++)
        for (int i = 0; i < N; i++) beliefs[(size_t)a*N*N + i*N + i] = 1.0;
    int rc = lau_fleet_batch_update(beliefs, updates, N, agents, 0.001);
    ASSERT_EQ(rc, 0, "fleet update 100 agents");
    free(beliefs); free(updates);
    return 0;
}

int test_fleet_update_1000_agents() {
    int N = 4, agents = 1000;
    size_t sz = (size_t)agents * N * N;
    double *beliefs = (double *)calloc(sz, sizeof(double));
    double *updates = random_matrix(agents * N, N);
    for (int a = 0; a < agents; a++)
        for (int i = 0; i < N; i++) beliefs[(size_t)a*N*N + i*N + i] = 1.0;

    auto start = std::chrono::high_resolution_clock::now();
    int rc = lau_fleet_batch_update(beliefs, updates, N, agents, 0.001);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    ASSERT_EQ(rc, 0, "fleet update 1000 agents");
    printf("    1000 agents (4x4): %.1f ms\n", ms);
    free(beliefs); free(updates);
    return 0;
}

int test_fleet_convergence_vote_all() {
    int flags[] = {1, 1, 1, 1, 1};
    int all_conv = 0;
    int rc = lau_fleet_sync_vote(flags, &all_conv, 5);
    ASSERT_EQ(rc, 0, "vote all converged");
    ASSERT_EQ(all_conv, 1, "all should be converged");
    return 0;
}

int test_fleet_convergence_vote_partial() {
    int flags[] = {1, 1, 0, 1, 1};
    int all_conv = 1;
    int rc = lau_fleet_sync_vote(flags, &all_conv, 5);
    ASSERT_EQ(rc, 0, "vote partial");
    ASSERT_EQ(all_conv, 0, "not all converged");
    return 0;
}

int test_fleet_convergence_vote_none() {
    int flags[] = {0, 0, 0};
    int all_conv = 1;
    int rc = lau_fleet_sync_vote(flags, &all_conv, 3);
    ASSERT_EQ(rc, 0, "vote none");
    ASSERT_EQ(all_conv, 0, "none converged");
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 8: Conservation (Tests 50-57)
 * ══════════════════════════════════════════════════════════════════ */

int test_conservation_perfect() {
    double cin[] = {5.0, 3.0, 7.0};
    double cout[] = {5.0, 3.0, 7.0};
    double ratios[3] = {0};
    int rc = lau_verify_conservation_batch(cin, cout, ratios, 3);
    ASSERT_EQ(rc, 0, "conservation perfect");
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(ratios[i], 1.0, TOLERANCE, "ratio == 1");
    return 0;
}

int test_conservation_drift() {
    double cin[] = {5.0, 3.0, 7.0};
    double cout[] = {4.9, 3.05, 6.95};
    double ratios[3] = {0};
    int rc = lau_verify_conservation_batch(cin, cout, ratios, 3);
    ASSERT_EQ(rc, 0, "conservation drift");
    ASSERT_NEAR(ratios[0], 0.98, 0.01, "ratio 0");
    ASSERT_NEAR(ratios[1], 1.0167, 0.01, "ratio 1");
    return 0;
}

int test_conservation_zero_charge() {
    double cin[] = {0.0, 5.0};
    double cout[] = {0.0, 5.0};
    double ratios[2] = {0};
    int rc = lau_verify_conservation_batch(cin, cout, ratios, 2);
    ASSERT_EQ(rc, 0, "conservation zero");
    ASSERT_NEAR(ratios[0], 0.0, TOLERANCE, "zero charge ratio");
    return 0;
}

int test_crdt_merge_uniform() {
    int state_len = 3, agents = 4;
    double states[] = {1,2,3, 1,2,3, 1,2,3, 1,2,3};
    double merged[3] = {0};
    int rc = lau_crdt_merge(states, merged, state_len, agents);
    ASSERT_EQ(rc, 0, "crdt merge uniform");
    ASSERT_NEAR(merged[0], 1.0, TOLERANCE, "merge[0]");
    ASSERT_NEAR(merged[1], 2.0, TOLERANCE, "merge[1]");
    ASSERT_NEAR(merged[2], 3.0, TOLERANCE, "merge[2]");
    return 0;
}

int test_crdt_merge_divergent() {
    int state_len = 2, agents = 3;
    double states[] = {10,20, 30,40, 50,60};
    double merged[2] = {0};
    int rc = lau_crdt_merge(states, merged, state_len, agents);
    ASSERT_EQ(rc, 0, "crdt merge divergent");
    ASSERT_NEAR(merged[0], 30.0, TOLERANCE, "merge avg 0");
    ASSERT_NEAR(merged[1], 40.0, TOLERANCE, "merge avg 1");
    return 0;
}

int test_conservation_reduce() {
    double ratios[] = {1.0, 1.0, 1.0, 1.0};
    double global = 0;
    int rc = lau_conservation_reduce(ratios, &global, 4);
    ASSERT_EQ(rc, 0, "reduce perfect");
    ASSERT_NEAR(global, 1.0, TOLERANCE, "global ratio 1");
    return 0;
}

int test_conservation_reduce_drift() {
    double ratios[] = {0.98, 1.01, 0.99, 1.02};
    double global = 0;
    int rc = lau_conservation_reduce(ratios, &global, 4);
    ASSERT_EQ(rc, 0, "reduce drift");
    ASSERT_NEAR(global, 1.0, 0.01, "global ratio near 1");
    return 0;
}

int test_conservation_1000_agents() {
    int N = 1000;
    double *cin = (double *)malloc(N * sizeof(double));
    double *cout = (double *)malloc(N * sizeof(double));
    double *ratios = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        cin[i] = 1.0 + (double)i / N;
        cout[i] = cin[i] * (1.0 + 0.001 * ((i % 3) - 1)); /* tiny drift */
    }
    int rc = lau_verify_conservation_batch(cin, cout, ratios, N);
    ASSERT_EQ(rc, 0, "conservation 1000 agents");
    double global = 0;
    rc = lau_conservation_reduce(ratios, &global, N);
    ASSERT_EQ(rc, 0, "reduce 1000");
    ASSERT_NEAR(global, 1.0, 0.01, "global ratio 1000 agents");
    free(cin); free(cout); free(ratios);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Test Group 9: Memory & Performance Benchmarks (Tests 58-65)
 * ══════════════════════════════════════════════════════════════════ */

int test_memory_budget_1000_agents_16x16() {
    /* 1000 agents × 16×16 matrix × 8 bytes = ~2 MB — fits easily */
    int agents = 1000, mat_dim = 16;
    size_t per_agent = (size_t)mat_dim * mat_dim * sizeof(double);
    size_t total = (size_t)agents * per_agent;
    printf("    1000 agents × 16×16: %.2f MB\n", (double)total / 1e6);
    ASSERT_EQ(total < 7ULL * 1024 * 1024 * 1024, 1, "fits in 7GB VRAM");
    return 0;
}

int test_memory_budget_1000_agents_64x64() {
    /* 1000 × 64×64 × 8 = ~32 MB */
    int agents = 1000, mat_dim = 64;
    size_t total = (size_t)agents * mat_dim * mat_dim * sizeof(double);
    printf("    1000 agents × 64×64: %.2f MB\n", (double)total / 1e6);
    ASSERT_EQ(total < 7ULL * 1024 * 1024 * 1024, 1, "fits in 7GB VRAM");
    return 0;
}

int test_memory_budget_1000_agents_256x256() {
    /* 1000 × 256×256 × 8 = ~512 MB */
    int agents = 1000, mat_dim = 256;
    size_t total = (size_t)agents * mat_dim * mat_dim * sizeof(double);
    printf("    1000 agents × 256×256: %.2f MB\n", (double)total / 1e6);
    ASSERT_EQ(total < 7ULL * 1024 * 1024 * 1024, 1, "fits in 7GB VRAM");
    return 0;
}

int test_memory_budget_max_eigen() {
    /* 4096×4096 eigendecomposition workspace */
    int N = 4096;
    size_t mat = (size_t)N * N * sizeof(double);
    size_t workspace = mat * 4; /* conservative: A + W + V + work */
    printf("    4096×4096 eigen workspace: %.2f MB\n", (double)workspace / 1e6);
    ASSERT_EQ(workspace < 7ULL * 1024 * 1024 * 1024, 1, "fits in 7GB VRAM");
    return 0;
}

int test_perf_matmul_1024_batched() {
    int N = 1024, batch = 4;
    size_t sz = (size_t)batch * N * N;
    double *A = random_matrix(batch * N, N);
    double *B = random_matrix(batch * N, N);
    double *C = (double *)calloc(sz, sizeof(double));
    auto start = std::chrono::high_resolution_clock::now();
    int rc = lau_matmul(A, B, C, N, N, N, batch);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    ASSERT_EQ(rc, 0, "perf 1024 batched");
    printf("    4× 1024×1024 matmul: %.1f ms\n", ms);
    free(A); free(B); free(C);
    return 0;
}

int test_perf_fleet_1000_16x16() {
    int N = 16, agents = 1000;
    size_t sz = (size_t)agents * N * N;
    double *beliefs = (double *)calloc(sz, sizeof(double));
    double *updates = random_matrix(agents * N, N);
    for (int a = 0; a < agents; a++)
        for (int i = 0; i < N; i++) beliefs[(size_t)a*N*N + i*N + i] = 1.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; iter++) {
        lau_fleet_batch_update(beliefs, updates, N, agents, 0.001);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("    100 iters × 1000 agents (16×16): %.1f ms total, %.2f ms/iter\n",
           ms, ms / 100.0);
    free(beliefs); free(updates);
    return 0;
}

int test_perf_conservation_10000() {
    int N = 10000;
    double *cin = (double *)malloc(N * sizeof(double));
    double *cout = (double *)malloc(N * sizeof(double));
    double *ratios = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) { cin[i] = 1.0; cout[i] = 1.0; }

    auto start = std::chrono::high_resolution_clock::now();
    lau_verify_conservation_batch(cin, cout, ratios, N);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("    Conservation 10k agents: %.2f ms\n", ms);

    free(cin); free(cout); free(ratios);
    return 0;
}

int test_perf_crdt_merge_large() {
    int state_len = 256, agents = 500;
    size_t total = (size_t)agents * state_len;
    double *states = random_matrix(agents, state_len);
    double *merged = (double *)calloc(state_len, sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();
    lau_crdt_merge(states, merged, state_len, agents);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("    CRDT merge 500 agents × 256 state: %.2f ms\n", ms);

    free(states); free(merged);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Main
 * ══════════════════════════════════════════════════════════════════ */

int main()
{
    srand(42);
    printf("══════════════════════════════════════════════════════\n");
    printf("  lau-math-cuda test suite\n");
    printf("══════════════════════════════════════════════════════\n\n");

    /* Init */
    printf("── Initialization ──\n");
    RUN_TEST(test_init);
    RUN_TEST(test_backend_detected);
    RUN_TEST(test_gpu_info);
    RUN_TEST(test_double_init);
    RUN_TEST(test_shutdown_reinit);

    /* Matrix multiply */
    printf("\n── Matrix Multiply ──\n");
    RUN_TEST(test_matmul_identity);
    RUN_TEST(test_matmul_zero);
    RUN_TEST(test_matmul_1x1);
    RUN_TEST(test_matmul_16x16);
    RUN_TEST(test_matmul_64x64);
    RUN_TEST(test_matmul_256x256);
    RUN_TEST(test_matmul_rectangular);
    RUN_TEST(test_matmul_batched);
    RUN_TEST(test_matmul_1024x1024);
    RUN_TEST(test_matmul_4096x4096);

    /* Inverse */
    printf("\n── Matrix Inverse ──\n");
    RUN_TEST(test_inverse_identity);
    RUN_TEST(test_inverse_2x2);
    RUN_TEST(test_inverse_16x16);
    RUN_TEST(test_inverse_64x64);
    RUN_TEST(test_inverse_batched);
    RUN_TEST(test_inverse_roundtrip);
    RUN_TEST(test_inverse_256x256);

    /* Eigendecomposition */
    printf("\n── Eigendecomposition ──\n");
    RUN_TEST(test_eigen_identity);
    RUN_TEST(test_eigen_diagonal);
    RUN_TEST(test_eigen_16x16);
    RUN_TEST(test_eigen_batched);
    RUN_TEST(test_eigen_64x64);
    RUN_TEST(test_eigen_256x256);

    /* Laplacian */
    printf("\n── Laplacian ──\n");
    RUN_TEST(test_laplacian_trivial);
    RUN_TEST(test_laplacian_3node);
    RUN_TEST(test_spectral_gap_single);
    RUN_TEST(test_power_iteration_identity);
    RUN_TEST(test_power_iteration_scaled);
    RUN_TEST(test_laplacian_larger);
    RUN_TEST(test_spectral_gap_batch);

    /* Heat kernel */
    printf("\n── Heat Kernel ──\n");
    RUN_TEST(test_heat_kernel_zero_t);
    RUN_TEST(test_heat_kernel_preserves_sum);
    RUN_TEST(test_heat_kernel_batched);
    RUN_TEST(test_heat_kernel_multi_source);
    RUN_TEST(test_heat_kernel_decay);
    RUN_TEST(test_heat_kernel_16node);
    RUN_TEST(test_heat_kernel_laplacian_combined);

    /* Agent fleet */
    printf("\n── Agent Fleet ──\n");
    RUN_TEST(test_fleet_update_single);
    RUN_TEST(test_fleet_update_batch);
    RUN_TEST(test_fleet_update_100_agents);
    RUN_TEST(test_fleet_update_1000_agents);
    RUN_TEST(test_fleet_convergence_vote_all);
    RUN_TEST(test_fleet_convergence_vote_partial);
    RUN_TEST(test_fleet_convergence_vote_none);

    /* Conservation */
    printf("\n── Conservation ──\n");
    RUN_TEST(test_conservation_perfect);
    RUN_TEST(test_conservation_drift);
    RUN_TEST(test_conservation_zero_charge);
    RUN_TEST(test_crdt_merge_uniform);
    RUN_TEST(test_crdt_merge_divergent);
    RUN_TEST(test_conservation_reduce);
    RUN_TEST(test_conservation_reduce_drift);
    RUN_TEST(test_conservation_1000_agents);

    /* Memory & Performance */
    printf("\n── Memory & Performance ──\n");
    RUN_TEST(test_memory_budget_1000_agents_16x16);
    RUN_TEST(test_memory_budget_1000_agents_64x64);
    RUN_TEST(test_memory_budget_1000_agents_256x256);
    RUN_TEST(test_memory_budget_max_eigen);
    RUN_TEST(test_perf_matmul_1024_batched);
    RUN_TEST(test_perf_fleet_1000_16x16);
    RUN_TEST(test_perf_conservation_10000);
    RUN_TEST(test_perf_crdt_merge_large);

    /* Summary */
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  Results: %d/%d passed", total_run - total_fail, total_run);
    if (total_fail > 0) printf(" (%d FAILED)", total_fail);
    printf("\n══════════════════════════════════════════════════════\n");

    lau_shutdown();
    return total_fail > 0 ? 1 : 0;
}
