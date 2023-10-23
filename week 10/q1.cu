#include <stdio.h>
#include <cuda_runtime.h>

#define FILTER_SIZE 3
#define ARRAY_SIZE 10

// Define the filter coefficients
__constant__ float filter[FILTER_SIZE];

// CUDA kernel to perform 1D convolution using constant memory
__global__ void convolutionKernel(const float* input, float* output, int arraySize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < arraySize) {
        float result = 0.0f;

        for (int i = 0; i < FILTER_SIZE; i++) {
            int idx = tid - FILTER_SIZE / 2 + i;
            if (idx >= 0 && idx < arraySize) {
                result += filter[i] * input[idx];
            }
        }

        output[tid] = result;
    }
}

int main() {
    // Initialize input data
    float hostInput[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        hostInput[i] = (float)i;
    }

    // Define the filter coefficients
    float hostFilter[FILTER_SIZE] = {1.0f, 2.0f, 1.0f};
    cudaMemcpyToSymbol(filter, hostFilter, FILTER_SIZE * sizeof(float));

    // Allocate device memory for input and output data
    float* deviceInput = NULL;
    float* deviceOutput = NULL;
    cudaMalloc((void**)&deviceInput, ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&deviceOutput, ARRAY_SIZE * sizeof(float));

    // Copy input data to the device
    cudaMemcpy(deviceInput, hostInput, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int blockSize = 256;
    int numBlocks = (ARRAY_SIZE + blockSize - 1) / blockSize;
    convolutionKernel<<<numBlocks, blockSize>>>(deviceInput, deviceOutput, ARRAY_SIZE);

    // Copy the result back to the host
    float hostOutput[ARRAY_SIZE];
    cudaMemcpy(hostOutput, deviceOutput, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("Output[%d] = %f\n", i, hostOutput[i]);
    }

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}
