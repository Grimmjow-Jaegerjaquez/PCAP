#include <stdio.h>
#include <cuda_runtime.h>

// Constants for sorting
__constant__ int d_N;
__constant__ int d_NumPhases;

// CUDA kernel to perform Odd-Even Transposition Sort
__global__ void oddEvenSort(int* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int phase = 0; phase < d_NumPhases; phase++) {
        int partner = tid ^ 1;
        if (tid % 2 == phase % 2) {
            if (partner < d_N) {
                if ((tid < d_N) && (data[tid] > data[partner])) {
                    // Swap elements
                    int temp = data[tid];
                    data[tid] = data[partner];
                    data[partner] = temp;
                }
            }
        }
    }
}

int main() {
    int N = 10, hostData[N];
    int* deviceData;

    // Initialize input data (you can replace this with your own data)
    for (int i = 0; i < N; i++) {
        hostData[i] = N - i;
    }

    for(int i = 0; i < N; i++){
        printf("%d ", hostData[i]);
    }
    printf("\n");

    // Allocate device memory for input data
    cudaMalloc((void**)&deviceData, N * sizeof(int));

    // Copy input data to the device
    cudaMemcpy(deviceData, hostData, N * sizeof(int), cudaMemcpyHostToDevice);

    // Set constants on the device
    int numPhases = N;
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_NumPhases, &numPhases, sizeof(int));

    // Launch the CUDA kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    oddEvenSort<<<numBlocks, blockSize>>>(deviceData);

    int data[N];
    cudaMemcpy(data, deviceData, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted output
    printf("Sorted Data: ");
    for (int i = N - 1; i >= 0; i--) {
        printf("%d ", data[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(deviceData);

    return 0;
}