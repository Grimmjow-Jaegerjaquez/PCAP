#include <stdio.h>

#define N 5 // Length of the vectors

__global__ void vectorAddBlock(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

__global__ void vectorAddThread(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

int main() {
    int *h_a, *h_b, *h_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory on the host
    h_a = (int*)malloc(N * sizeof(int));
    h_b = (int*)malloc(N * sizeof(int));
    h_c = (int*)malloc(N * sizeof(int));

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Part A: Block size as N
    vectorAddBlock<<<1, N>>>(d_a, d_b, d_c);

    // Part B: N threads
    int numBlocks = (N + 255) / 256; // Use enough blocks to cover N threads
    vectorAddThread<<<numBlocks, 256>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on the host
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}