#include <stdio.h>

const int N = 3;  // Size of the input and output arrays
const int M = 3;  // Size of the mask array

// Kernel for 2D convolution
__global__ void convolution2D(float* input, float* mask, float* output, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float result = 0.0f;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                int input_row = row + i - (M / 2);
                int input_col = col + j - (M / 2);
                if (input_row >= 0 && input_row < n && input_col >= 0 && input_col < n) {
                    result += input[input_row * n + input_col] * mask[i * M + j];
                }
            }
        }
        output[row * n + col] = result;
    }
}

int main() {
    float h_input[][N] = {{1, 2, 3}, {4, 5, 6} , {7, 8, 9}}; 
    float h_mask[][N] = {{-1, -2, -1} ,{0, 0, 0}, {1, 2, 1}}, *h_output;  // Host arrays
    float *d_input, *d_mask, *d_output;  // Device arrays
    int array_size = N * N * sizeof(float);
    int mask_size = M * M * sizeof(float);

    // Allocate memory on the host
    h_output = (float*)malloc(array_size);

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, array_size);
    cudaMalloc((void**)&d_mask, mask_size);
    cudaMalloc((void**)&d_output, array_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid(N, N);
    dim3 dimBlock(1, 1);

    // Launch the convolution kernel
    convolution2D<<<dimGrid, dimBlock>>>(d_input, d_mask, d_output, N);

    // Copy the result array from device to host
    cudaMemcpy(h_output, d_output, array_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_mask);
    free(h_output);

    // Process and use the output array h_output
    // ...

    return 0;
}