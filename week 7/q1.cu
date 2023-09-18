#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Matrix size

// Kernel to add two matrices where each row is computed by one thread
__global__ void addMatrixRow(int *a, int *b, int *c) {
    int tid = threadIdx.x;  // Thread index
    c[tid] = a[tid] + b[tid];
}

// Kernel to add two matrices where each column is computed by one thread
__global__ void addMatrixColumn(int *a, int *b, int *c) {
    int row = blockIdx.y;
    int col = threadIdx.x;
    int index = row * N + col;
    c[index] = a[index] + b[index];
}

// Kernel to add two matrices where each element is computed by one thread
__global__ void addMatrixElement(int *a, int *b, int *c) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int index = row * N + col;
    c[index] = a[index] + b[index];
}

int main() {
    int a[N][N], b[N][N], c[N][N];  // Input and output matrices
    int *dev_a, *dev_b, *dev_c;    // Device pointers

    // Allocate memory on the GPU
    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * N * sizeof(int));

    // Initialize matrices a and b with sample values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 1;
            b[i][j] = 2;
        }
    }

    // Copy matrices a and b from host to device
    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Specify the grid and block dimensions for the kernel launch
    dim3 dimGrid(N, N);   // Grid dimensions for addMatrixElement
    dim3 dimBlock(N, N);  // Block dimensions for addMatrixRow and addMatrixColumn

    // Launch the kernel to add matrices using the specified method
    addMatrixRow<<<1, dimBlock>>>(dev_a, dev_b, dev_c);
    addMatrixColumn<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);
    addMatrixElement<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);

    // Copy the result matrix c from device to host
    cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix c
    printf("Resultant Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}