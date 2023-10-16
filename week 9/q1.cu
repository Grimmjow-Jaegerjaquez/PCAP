#include <stdio.h>

const int N = 4;  // Matrix size (N x N)
const int BLOCK_SIZE = 2;

__global__ void matrixMul(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int h_A[N][N], h_B[N][N], h_C[N][N];  // Host matrices
    int *d_A, *d_B, *d_C;  // Device matrices
    int matrix_size = N * N * sizeof(int);
    

    // Initialize matrices h_A and h_B
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            h_A[i][j] = i;
            h_B[i][j] = i + j;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch the kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory

    // Perform further processing with the result matrix h_C
    // ...

    return 0;
}

