#include <stdio.h>
#include <stdlib.h>

// Define matrix and vector dimensions
#define N 4

// Kernel for sparse matrix-vector multiplication
__global__ void sparseMatVecMul(int* row_ptr, int* col_idx, float* values, float* x, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        float sum = 0.0f;
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        
        for (int i = start; i < end; i++) {
            int j = col_idx[i];
            sum += values[i] * x[j];
        }
        
        y[tid] = sum;
    }
}

int main() {
    // Define the CSR matrix
    int row_ptr[N + 1] = {0, 2, 4, 7, 9};
    int col_idx[] = {0, 2, 1, 3, 0, 1, 2, 0, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    
    // Define the input vector and output vector
    float x[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[N];
    
    // Allocate GPU memory
    int* d_row_ptr, *d_col_idx;
    float* d_values, *d_x, *d_y;
    
    cudaMalloc((void**)&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, sizeof(col_idx));
    cudaMalloc((void**)&d_values, sizeof(values));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, sizeof(col_idx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(values), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Perform matrix-vector multiplication on GPU
    sparseMatVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y);
    
    // Copy the result from device to host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    // Print the result
    for (int i = 0; i < N; i++) {
        printf("y[%d] = %f\n", i, y[i]);
    }
    
    return 0;
}