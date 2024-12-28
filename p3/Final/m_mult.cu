#include <stdio.h>


// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
    void dot_fwd_cublas(float *act, float *weights, float *zout, int Batch, int In, int Out);
    void dot_bwd_cublas(float *deltas, float *weights, float *bwdout, int Batch, int In, int Out);
    void dot_grd_cublas(float *act, float *deltas, float *grd, int Batch, int In, int Out);
}


// Error checking macro for CUDA operations
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Error checking macro for cuBLAS operations
#define cublasCheckError(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line, bool abort=true) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", status, file, line);
        if (abort) exit(status);
    }
}

extern "C" {

void dot_fwd_cublas(float *act, float *weights, float *zout, int Batch, int In, int Out) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_act, *d_weights, *d_zout;
    cudaMalloc(&d_act, Batch * In * sizeof(float));
    cudaMalloc(&d_weights, Out * In * sizeof(float));
    cudaMalloc(&d_zout, Batch * Out * sizeof(float));

    cudaMemcpy(d_act, act, Batch * In * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, Out * In * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    // Since cuBLAS assumes column-major storage, we adjust the operation to get the desired result
    // when input matrices are actually in row-major order. This involves swapping the order of the matrices
    // and using the correct transposition flags to effectively perform (B^T * A^T)^T = A * B^T in row-major sense
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                Out, Batch, In, 
                &alpha, 
                d_weights, In, // Note the swapped order and transposed B
                d_act, In, 
                &beta, 
                d_zout, Out);

    cudaMemcpy(zout, d_zout, Batch * Out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_act);
    cudaFree(d_weights);
    cudaFree(d_zout);
    cublasDestroy(handle);
}

void dot_bwd_cublas(float *deltas, float *weights, float *bwdout, int Batch, int In, int Out) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_deltas, *d_weights, *d_bwdout;
    cudaMalloc(&d_deltas, Batch * Out * sizeof(float));
    cudaMalloc(&d_weights, Out * In * sizeof(float));
    cudaMalloc(&d_bwdout, Batch * In * sizeof(float));

    cudaMemcpy(d_deltas, deltas, Batch * Out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, Out * In * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication for the backward pass
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                In, Batch, Out, 
                &alpha, 
                d_weights, In, 
                d_deltas, Out, 
                &beta, 
                d_bwdout, In);

    cudaMemcpy(bwdout, d_bwdout, Batch * In * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_deltas);
    cudaFree(d_weights);
    cudaFree(d_bwdout);
    cublasDestroy(handle);
}

void dot_grd_cublas(float *act, float *del, float *grd, int Batch, int In, int Out) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_act, *d_del, *d_grd;
    cudaMalloc(&d_act, Batch * In * sizeof(float));
    cudaMalloc(&d_del, Batch * Out * sizeof(float));
    cudaMalloc(&d_grd, Out * In * sizeof(float));

    cudaMemcpy(d_act, act, Batch * In * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_del, del, Batch * Out * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Adjust for row-major order logic using column-major order functions
    // Here, we transpose the operation to match the row-major logic applied in CBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                In, Out, Batch,
                &alpha,
                d_act, In, // Leading dimension reflects row length in row-major order
                d_del, Out, // Leading dimension reflects row length in row-major order
                &beta,
                d_grd, In); // LD of the output matrix in row-major logic

    cudaMemcpy(grd, d_grd, Out * In * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_act);
    cudaFree(d_del);
    cudaFree(d_grd);
    cublasDestroy(handle);
}

}
