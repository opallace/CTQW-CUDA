#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

using namespace std;

int main(){
	/* Dimensão da matriz Hermitiana/Laplaciana
	 */
	int n = 5000;

	/* Tempo
	 */
	float t = 1;

	/* Matriz Hermitiana/Laplaciana com
	 * dimensões NxN e seus valores zerados.
	 */
	vector<cuFloatComplex> L(n * n, make_cuFloatComplex(.0f, .0f));

	/* Define os valores da matriz Hermitiana/Laplaciana
	 * de forma simétrica correspondente as adjacências
	 * de um grafo.
	 */
	for(int i = 0; i < n; i++){
		for(int j = i + 1; j < n; j++){
			int value = rand() % 2 - 1;
			L[i * n + j] = make_cuFloatComplex(value, .0f);
			L[j * n + i] = make_cuFloatComplex(value, .0f);
		}
	}

	/* Define os valores da diagonal principal
	 * da matriz Hermitiana/Laplaciana correspondente
	 * ao grau do vértice.
	 * 
	 * Da para otimizar essa definição sem ter que percorrer
	 * todos os valores, apenas os elementos acima ou abaixo
	 * da diagonal
	 */
	for(int i = 0; i < n; i++){
		float degree = .0f;

		for(int j = 0; j < n; j++){
			degree += cuCabsf(L[i * n + j]);
		}
		
		L[i * n + i] = make_cuFloatComplex(degree, .0f);
	}

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

    for(int i = 0; i < n; i++){
        float re   = cuCrealf(psi_t[i]);
        float im   = cuCimagf(psi_t[i]);
        float prob = re * re + im * im;

        printf("  node %d: (%.4f, %.4f), |ψ|² = %.4f\n", i, re, im, prob);
    }

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
