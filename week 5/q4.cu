#include <stdio.h>
#include <math.h>

#define N 5 // Length of the array
#define THREADS_PER_BLOCK 256

__global__ void computeSine(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = sinf(input[tid]);
    }
}

int main() {
    float *h_input, *h_output; // Host arrays
    float *d_input, *d_output; // Device arrays

    // Allocate memory on the host
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Initialize host input array with angles in radians
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i) * (3.14159f / 180.0f); // Convert degrees to radians
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel
    computeSine<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        printf("sin(%.2f radians) = %.6f\n", h_input[i], h_output[i]);
    }

    // Free memory on the device
    cudaFree(d_input);
    cudaFree(d_output);

    // Free memory on the host
    free(h_input);
    free(h_output);

    return 0;
}