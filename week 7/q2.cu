#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Matrix size

// Kernel to multiply two matrices where each row is computed by one thread
__global__ void multiplyMatrixRow(int *a, int *b, int *c) {
    int row = threadIdx.x;  // Thread index
    for (int col = 0; col < N; col++) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// Kernel to multiply two matrices where each column is computed by one thread
__global__ void multiplyMatrixColumn(int *a, int *b, int *c) {
    int col = threadIdx.x;  // Thread index
    for (int row = 0; row < N; row++) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// Kernel to multiply two matrices where each element is computed by one thread
__global__ void multiplyMatrixElement(int *a, int *b, int *c) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int sum = 0;
    for (int k = 0; k < N; k++) {
        sum += a[row * N + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
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
            a[i][j] = 2;
            b[i][j] = 2;
        }
    }

    // Copy matrices a and b from host to device
    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Specify the block dimensions for the kernel launch
    dim3 dimBlock(N, 1);  // Block dimensions for row-wise and column-wise computation
    dim3 dimGrid(N, N);   // Grid dimensions for element-wise computation

    // Launch the kernel to multiply matrices using the specified method
    multiplyMatrixRow<<<1, dimBlock>>>(dev_a, dev_b, dev_c);
    multiplyMatrixColumn<<<1, dimBlock>>>(dev_a, dev_b, dev_c);
    multiplyMatrixElement<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);

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