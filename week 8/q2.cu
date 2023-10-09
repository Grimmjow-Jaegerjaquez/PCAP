#include <stdio.h>
#include <stdlib.h>

#define M 4 // Number of rows
#define N 4 // Number of columns

__global__ void powerRows(float* matrix, int rows, int cols, int currentRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < cols) {
        int index = currentRow * cols + tid;
        float element = matrix[index];
        
        // Compute the power based on the current row
        matrix[index] = powf(element, currentRow + 1); // +1 to start with 1st row
    }
}

int main() {
    float* h_matrix; // Host matrix
    float* d_matrix; // Device matrix
    
    // Allocate memory for the host matrix
    h_matrix = (float*)malloc(M * N * sizeof(float));
    
    // Initialize the host matrix (you can read from input instead)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_matrix[i * N + j] = (float)(i * N + j + 1); // Fill the matrix with increasing numbers
        }
    }
    
    // Allocate memory for the device matrix
    cudaMalloc((void**)&d_matrix, M * N * sizeof(float));
    
    // Copy the host matrix to the device
    cudaMemcpy(d_matrix, h_matrix, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel for each row
    for (int row = 0; row < M; row++) {
        powerRows<<<1, N>>>(d_matrix, M, N, row);
        cudaDeviceSynchronize();
    }
    
    // Copy the updated matrix back to the host
    cudaMemcpy(h_matrix, d_matrix, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the updated matrix
    printf("Updated Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", h_matrix[i * N + j]);
        }
        printf("\n");
    }
    
    // Free allocated memory
    free(h_matrix);
    cudaFree(d_matrix);
    
    return 0;
}