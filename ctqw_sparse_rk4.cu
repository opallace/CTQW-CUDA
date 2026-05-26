#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusparse.h>

using namespace std;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CUSPARSE_CHECK(call)                                               \
    do {                                                                   \
        cusparseStatus_t status = call;                                    \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",               \
                    __FILE__, __LINE__, static_cast<int>(status));         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

struct Edge {
    int u;
    int v;
    float w;
};

struct CsrMatrixReal {
    int n;
    vector<int> row_offsets;
    vector<int> col_indices;
    vector<float> values;
};

struct CsrMatrixComplex {
    int n;
    int nnz;
    vector<int> row_offsets;
    vector<int> col_indices;
    vector<cuComplex> values;
};

CsrMatrixReal build_laplacian_csr(int n, const vector<Edge>& edges) {
    vector<vector<pair<int, float>>> rows(n);
    vector<float> degree(n, 0.0f);

    for (const auto& e : edges) {
        float w = e.w;

        degree[e.u] += std::abs(w);
        degree[e.v] += std::abs(w);

        // Laplaciano ponderado:
        // L_uv = -w
        // L_vu = -w
        rows[e.u].push_back({e.v, -w});
        rows[e.v].push_back({e.u, -w});
    }

    // Diagonal:
    // L_ii = grau(i)
    for (int i = 0; i < n; i++) {
        rows[i].push_back({i, degree[i]});
    }

    CsrMatrixReal csr;
    csr.n = n;
    csr.row_offsets.resize(n + 1);

    int nnz = 0;

    for (int i = 0; i < n; i++) {
        auto& row = rows[i];

        sort(row.begin(), row.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );

        csr.row_offsets[i] = nnz;

        for (const auto& [col, value] : row) {
            csr.col_indices.push_back(col);
            csr.values.push_back(value);
            nnz++;
        }
    }

    csr.row_offsets[n] = nnz;

    return csr;
}

CsrMatrixComplex to_complex_csr(const CsrMatrixReal& real) {
    CsrMatrixComplex complex_csr;

    complex_csr.n = real.n;
    complex_csr.nnz = static_cast<int>(real.values.size());
    complex_csr.row_offsets = real.row_offsets;
    complex_csr.col_indices = real.col_indices;
    complex_csr.values.resize(real.values.size());

    for (size_t i = 0; i < real.values.size(); i++) {
        complex_csr.values[i] = make_cuFloatComplex(real.values[i], 0.0f);
    }

    return complex_csr;
}

__global__
void compute_ctqw_derivative_kernel(
    const cuComplex* lpsi,
    cuComplex* k,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    float a = cuCrealf(lpsi[i]);
    float b = cuCimagf(lpsi[i]);

    // f(psi) = -i L psi
    //
    // Se Lpsi = a + ib, então:
    // -i(a + ib) = b - ia
    k[i] = make_cuFloatComplex(b, -a);
}

__global__
void make_rk4_temp_kernel(
    const cuComplex* psi,
    const cuComplex* k,
    cuComplex* temp,
    int n,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    cuComplex scaled_k = make_cuFloatComplex(
        scale * cuCrealf(k[i]),
        scale * cuCimagf(k[i])
    );

    temp[i] = cuCaddf(psi[i], scaled_k);
}

__global__
void rk4_update_kernel(
    cuComplex* psi,
    const cuComplex* k1,
    const cuComplex* k2,
    const cuComplex* k3,
    const cuComplex* k4,
    int n,
    float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    float re =
        cuCrealf(k1[i])
        + 2.0f * cuCrealf(k2[i])
        + 2.0f * cuCrealf(k3[i])
        + cuCrealf(k4[i]);

    float im =
        cuCimagf(k1[i])
        + 2.0f * cuCimagf(k2[i])
        + 2.0f * cuCimagf(k3[i])
        + cuCimagf(k4[i]);

    cuComplex delta = make_cuFloatComplex(
        (dt / 6.0f) * re,
        (dt / 6.0f) * im
    );

    psi[i] = cuCaddf(psi[i], delta);
}

void compute_derivative(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t matL,
    cusparseDnVecDescr_t vecInput,
    cusparseDnVecDescr_t vecLPsi,
    const cuComplex* d_lpsi,
    cuComplex* d_k,
    void* d_buffer,
    int n
) {
    cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    cuComplex beta  = make_cuFloatComplex(0.0f, 0.0f);

    // d_lpsi = L * input
    CUSPARSE_CHECK(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matL,
        vecInput,
        &beta,
        vecLPsi,
        CUDA_C_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_buffer
    ));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // d_k = -i * d_lpsi
    compute_ctqw_derivative_kernel<<<blocks, threads>>>(
        d_lpsi,
        d_k,
        n
    );

    CUDA_CHECK(cudaGetLastError());
}

int main() {
    int n = 5000;
    float t = 1.0f;

    // Número de passos temporais.
    // RK4 custa 4 SpMVs por passo.
    int steps = 1000;
    float dt = t / static_cast<float>(steps);

    vector<Edge> edges;

    // Probabilidade de aresta.
    // Para n = 5000 e p = 0.01, o número esperado de arestas é ~125k.
    // Para ~12.5k arestas, use p = 0.001f.
    float p = 0.001f;

    mt19937 rng(1234);
    uniform_real_distribution<float> prob(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (prob(rng) < p) {
                edges.push_back({i, j, 1.0f});
            }
        }
    }

    cout << "n = " << n << "\n";
    cout << "edges = " << edges.size() << "\n";

    CsrMatrixComplex csr = to_complex_csr(
        build_laplacian_csr(n, edges)
    );

    int nnz = csr.nnz;

    cout << "nnz = " << nnz << "\n";

    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    cuComplex* d_values = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, sizeof(int) * (n + 1)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&d_values, sizeof(cuComplex) * nnz));

    CUDA_CHECK(cudaMemcpy(
        d_row_offsets,
        csr.row_offsets.data(),
        sizeof(int) * (n + 1),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_col_indices,
        csr.col_indices.data(),
        sizeof(int) * nnz,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_values,
        csr.values.data(),
        sizeof(cuComplex) * nnz,
        cudaMemcpyHostToDevice
    ));

    // Estado inicial psi_0
    vector<cuComplex> psi(n, make_cuFloatComplex(0.0f, 0.0f));
    psi[0] = make_cuFloatComplex(1.0f, 0.0f);

    cuComplex* d_psi = nullptr;
    cuComplex* d_lpsi = nullptr;

    cuComplex* d_k1 = nullptr;
    cuComplex* d_k2 = nullptr;
    cuComplex* d_k3 = nullptr;
    cuComplex* d_k4 = nullptr;

    cuComplex* d_temp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_psi, sizeof(cuComplex) * n));
    CUDA_CHECK(cudaMalloc(&d_lpsi, sizeof(cuComplex) * n));

    CUDA_CHECK(cudaMalloc(&d_k1, sizeof(cuComplex) * n));
    CUDA_CHECK(cudaMalloc(&d_k2, sizeof(cuComplex) * n));
    CUDA_CHECK(cudaMalloc(&d_k3, sizeof(cuComplex) * n));
    CUDA_CHECK(cudaMalloc(&d_k4, sizeof(cuComplex) * n));

    CUDA_CHECK(cudaMalloc(&d_temp, sizeof(cuComplex) * n));

    CUDA_CHECK(cudaMemcpy(
        d_psi,
        psi.data(),
        sizeof(cuComplex) * n,
        cudaMemcpyHostToDevice
    ));

    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matL = nullptr;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matL,
        n,
        n,
        nnz,
        d_row_offsets,
        d_col_indices,
        d_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_32F
    ));

    cusparseDnVecDescr_t vecPsi = nullptr;
    cusparseDnVecDescr_t vecTemp = nullptr;
    cusparseDnVecDescr_t vecLPsi = nullptr;

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vecPsi,
        n,
        d_psi,
        CUDA_C_32F
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vecTemp,
        n,
        d_temp,
        CUDA_C_32F
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vecLPsi,
        n,
        d_lpsi,
        CUDA_C_32F
    ));

    cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    cuComplex beta  = make_cuFloatComplex(0.0f, 0.0f);

    size_t buffer_size = 0;
    void* d_buffer = nullptr;

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matL,
        vecPsi,
        &beta,
        vecLPsi,
        CUDA_C_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size
    ));

    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int step = 0; step < steps; step++) {
        // k1 = f(psi)
        compute_derivative(
            handle,
            matL,
            vecPsi,
            vecLPsi,
            d_lpsi,
            d_k1,
            d_buffer,
            n
        );

        // temp = psi + dt/2 * k1
        make_rk4_temp_kernel<<<blocks, threads>>>(
            d_psi,
            d_k1,
            d_temp,
            n,
            0.5f * dt
        );
        CUDA_CHECK(cudaGetLastError());

        // k2 = f(temp)
        compute_derivative(
            handle,
            matL,
            vecTemp,
            vecLPsi,
            d_lpsi,
            d_k2,
            d_buffer,
            n
        );

        // temp = psi + dt/2 * k2
        make_rk4_temp_kernel<<<blocks, threads>>>(
            d_psi,
            d_k2,
            d_temp,
            n,
            0.5f * dt
        );
        CUDA_CHECK(cudaGetLastError());

        // k3 = f(temp)
        compute_derivative(
            handle,
            matL,
            vecTemp,
            vecLPsi,
            d_lpsi,
            d_k3,
            d_buffer,
            n
        );

        // temp = psi + dt * k3
        make_rk4_temp_kernel<<<blocks, threads>>>(
            d_psi,
            d_k3,
            d_temp,
            n,
            dt
        );
        CUDA_CHECK(cudaGetLastError());

        // k4 = f(temp)
        compute_derivative(
            handle,
            matL,
            vecTemp,
            vecLPsi,
            d_lpsi,
            d_k4,
            d_buffer,
            n
        );

        // psi = psi + dt/6 * (k1 + 2k2 + 2k3 + k4)
        rk4_update_kernel<<<blocks, threads>>>(
            d_psi,
            d_k1,
            d_k2,
            d_k3,
            d_k4,
            n,
            dt
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        psi.data(),
        d_psi,
        sizeof(cuComplex) * n,
        cudaMemcpyDeviceToHost
    ));

    printf("ψ(%.3f), usando Laplaciano esparso + RK4:\n", t);

    // float total_prob = 0.0f;

    // for (int i = 0; i < n; i++) {
    //     float re = cuCrealf(psi[i]);
    //     float im = cuCimagf(psi[i]);
    //     float p_i = re * re + im * im;

    //     total_prob += p_i;

	// 	printf(
	// 		"  node %d: (%.6f, %.6f), |ψ|² = %.6f\n",
	// 		i,
	// 		re,
	// 		im,
	// 		p_i
	// 	);
    // }

    // printf("Total probability ≈ %.6f\n", total_prob);

    // Liberação
    if (d_buffer) {
        cudaFree(d_buffer);
    }

    cusparseDestroyDnVec(vecPsi);
    cusparseDestroyDnVec(vecTemp);
    cusparseDestroyDnVec(vecLPsi);

    cusparseDestroySpMat(matL);
    cusparseDestroy(handle);

    cudaFree(d_temp);

    cudaFree(d_k4);
    cudaFree(d_k3);
    cudaFree(d_k2);
    cudaFree(d_k1);

    cudaFree(d_lpsi);
    cudaFree(d_psi);

    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_offsets);

    return 0;
}