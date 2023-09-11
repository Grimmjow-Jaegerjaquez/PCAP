#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_STRING_LENGTH 1024

__global__ void generateRS(char* S, char* RS, int stringLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < stringLength) {
        RS[tid] = S[tid];
        RS[tid + stringLength] = S[tid];
        RS[tid + 2 * stringLength] = S[tid];
    }
}

int main() {
    char S[MAX_STRING_LENGTH] = "PCAP"; // Input string
    char RS[3 * MAX_STRING_LENGTH]; // Output string
    int stringLength = strlen(S);

    char* d_S;
    char* d_RS;

    cudaMalloc((void**)&d_S, stringLength * sizeof(char));
    cudaMalloc((void**)&d_RS, 3 * stringLength * sizeof(char));

    cudaMemcpy(d_S, S, stringLength * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (stringLength + blockSize - 1) / blockSize;

    generateRS<<<gridSize, blockSize>>>(d_S, d_RS, stringLength);

    cudaMemcpy(RS, d_RS, 3 * stringLength * sizeof(char), cudaMemcpyDeviceToHost);

    RS[3 * stringLength] = '\0'; // Null-terminate the output string

    printf("Input string S: %s\n", S);
    printf("Output string RS: %s\n", RS);

    cudaFree(d_S);
    cudaFree(d_RS);

    return 0;
}