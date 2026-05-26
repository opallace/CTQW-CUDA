#include <fstream>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

using namespace std;

struct Edge {
    int u;
    int v;
    float w;
};

inline int idx(int row, int col, int n) {
    // Layout column-major, compatível com cuSolver/cuBLAS.
    return row + col * n;
}

std::vector<Edge> read_edges_from_file(const std::string& filename, int& n) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Não foi possível abrir o arquivo: " + filename);
    }

    std::vector<Edge> edges;

    int u;
    int v;
    float w;

    int max_vertex = -1;

    while (file >> u >> v >> w) {
        edges.push_back({u, v, w});

        max_vertex = std::max(max_vertex, u);
        max_vertex = std::max(max_vertex, v);
    }

    n = max_vertex + 1;

    return edges;
}

std::vector<cuFloatComplex> build_dense_laplacian_from_edges(
    int n,
    const std::vector<Edge>& edges
) {
    std::vector<cuFloatComplex> L(
        n * n,
        make_cuFloatComplex(0.0f, 0.0f)
    );

    for (const auto& edge : edges) {
        int u = edge.u;
        int v = edge.v;
        float w = edge.w;

        // Off-diagonal do Laplaciano: L_uv = L_vu = -w
        L[idx(u, v, n)] = make_cuFloatComplex(-w, 0.0f);
        L[idx(v, u, n)] = make_cuFloatComplex(-w, 0.0f);

        // Diagonal: grau ponderado
        float degree_u = cuCrealf(L[idx(u, u, n)]) + std::abs(w);
        float degree_v = cuCrealf(L[idx(v, v, n)]) + std::abs(w);

        L[idx(u, u, n)] = make_cuFloatComplex(degree_u, 0.0f);
        L[idx(v, v, n)] = make_cuFloatComplex(degree_v, 0.0f);
    }

    return L;
}

int main(){
	/* Dimensão da matriz Hermitiana/Laplaciana
	 */
	int n = 5000;

	/* Tempo
	 */
	float t = 1;

	std::string filename = "output_ctqw2.txt";

    std::vector<Edge> edges;
    {
        int inferred_n = 0;
        edges = read_edges_from_file("output_ctqw2.txt", inferred_n);

        if (inferred_n > n) {
            throw std::runtime_error("O arquivo possui vértices com índice >= 5000.");
        }
    }


    std::cout << "n = " << n << "\n";
    std::cout << "edges = " << edges.size() << "\n";

    std::vector<cuFloatComplex> L = build_dense_laplacian_from_edges(n, edges);

	/* Aloca espaço na memória da placa de video para
	 * armazenar os autovalores e autovetores calculados.
	 */
	cuComplex *dEigenvectors = nullptr;
	cuComplex *dWork         = nullptr; 
	float *dEigenvalues      = nullptr;
	int   *dInfo             = nullptr;

	cudaMalloc(&dEigenvectors, sizeof(cuComplex) * n * n);
  	cudaMalloc(&dEigenvalues, sizeof(float) * n);
  	cudaMalloc(&dInfo, sizeof(int));
  	cudaMemcpy(dEigenvectors, L.data(), sizeof(cuComplex) * n * n, cudaMemcpyHostToDevice);

	cusolverDnHandle_t handle_cusolver = nullptr;
	cusolverDnCreate(&handle_cusolver);

	/* Calcula espaço necessário na memória da placa 
	 * de video para a realização dos calculos.
	 */
	int lwork = 0;
	cusolverDnCheevd_bufferSize(
		handle_cusolver,
		CUSOLVER_EIG_MODE_VECTOR,
		CUBLAS_FILL_MODE_LOWER,
		n,
		dEigenvectors,
		n,
		dEigenvalues,
		&lwork
	);

	/* Aloca espaço na memória da placa de vídeo para
	 * a realização dos calculos.
	 */
	cudaMalloc(&dWork, sizeof(cuComplex)*lwork);
	
	/* Realiza o calculo dos autovalores e autovetores 
	 * da matriz Hermitiana/Laplaciana.
	 */
	cusolverDnCheevd(
		handle_cusolver,
		CUSOLVER_EIG_MODE_VECTOR,
		CUBLAS_FILL_MODE_LOWER,
		n,
		dEigenvectors,
		n,
		dEigenvalues,
		dWork,
		lwork,
		dInfo
	);

	int info = 0;
  	cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost);

	if (info != 0) {
    	fprintf(stderr, "Cheevd falhou: info=%d\n", info);
    	return 1;
  	}

	/* Copia os autovalores e autovetores calculados da memória
	 * da placa de vídeo para a memória RAM. 
	 */
	vector<float> hEigenvalues(n);
	vector<cuComplex> hEigenvectors(n * n);
    cudaMemcpy(hEigenvalues.data(), dEigenvalues, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(hEigenvectors.data(), dEigenvectors, sizeof(cuComplex) * n * n, cudaMemcpyDeviceToHost);

	/* Aloca algumas variavéis para calculo da evolução temporal
	 */
	cublasHandle_t handle_cublas = nullptr;
	cublasCreate(&handle_cublas);

	vector<cuComplex> psi_t(n, make_cuFloatComplex(.0f, .0f));
	vector<cuComplex> psi_0(n, make_cuFloatComplex(.0f, .0f));
	psi_0[0] = make_cuFloatComplex(1.f, .0f);

	cuComplex *dpsi_t = nullptr;
	cuComplex *dpsi_0 = nullptr;
	
	cudaMalloc(&dpsi_t, sizeof(cuComplex) * n);
  	cudaMalloc(&dpsi_0, sizeof(cuComplex) * n);

	cudaMemcpy(dpsi_t, psi_t.data(), sizeof(cuComplex) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dpsi_0, psi_0.data(), sizeof(cuComplex) * n, cudaMemcpyHostToDevice);


	for(int k = 0; k < n; k++){
		cuComplex* eigenvector_k = dEigenvectors + k * n;
		cuComplex dot;
		
		/* Calcula e^{-i λ_k t}.
		 */
		float theta = -hEigenvalues[k] * t;
		cuFloatComplex phase = make_cuFloatComplex(cosf(theta), sinf(theta));

		/* Calcula <Ø_k|ψ_0>.
		 */
		cublasCdotc_v2(handle_cublas, n, eigenvector_k, 1, dpsi_0, 1, &dot);

		/* Calcula e^{-i λ_k t} <Ø_k|ψ_0>.
		 */
		cuComplex coeff = cuCmulf(dot, phase);

		/* Calcula e^{-i λ_k t} <Ø_k|ψ_0> |Ø_k> e soma o resultado
		 * ao ψ(t).
		 */
		cublasCaxpy_v2(handle_cublas, n, &coeff, eigenvector_k, 1, dpsi_t, 1);
	}

    cudaMemcpy(psi_t.data(), dpsi_t, sizeof(cuComplex)*n, cudaMemcpyDeviceToHost);

    printf("ψ(%.3f):\n", t);

    // for(int i = 0; i < n; i++){
    //     float re   = cuCrealf(psi_t[i]);
    //     float im   = cuCimagf(psi_t[i]);
    //     float prob = re * re + im * im;

    //     printf("  node %d: (%.4f, %.4f), |ψ|² = %.4f\n", i, re, im, prob);
    // }

	/* Libera espaço de memória alocado durante
	 * a execução do programa.
	 */
	cusolverDnDestroy(handle_cusolver);
	cublasDestroy(handle_cublas);
	cudaFree(dpsi_t); 
	cudaFree(dpsi_0); 
	cudaFree(dWork); 
	cudaFree(dInfo); 
	cudaFree(dEigenvectors); 
	cudaFree(dEigenvalues);
	
	return 0;
}