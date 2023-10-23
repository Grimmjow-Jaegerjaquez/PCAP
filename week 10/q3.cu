#include <stdio.h>
#include <cuda_runtime.h>

# define N 10

__global__ void mergeSort(int* data, int* temp, int left, int right) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= left && tid < right) {
        int mid = (left + right) / 2;
        int i = left;
        int j = mid;
        int k = left;

        while (i < mid && j < right) {
            if (data[i] < data[j]) {
                temp[k] = data[i];
                i++;
            } else {
                temp[k] = data[j];
                j++;
            }
            k++;
        }

        while (i < mid) {
            temp[k] = data[i];
            i++;
            k++;
        }

        while (j < right) {
            temp[k] = data[j];
            j++;
            k++;
        }

        for (int x = left; x < right; x++) {
            data[x] = temp[x];
        }
    }
}

int main() {
    int hostData[N];
    int* deviceData;
    int* deviceTemp;

    // Initialize input data (you can replace this with your own data)
    for (int i = 0; i < N; i++) {
        hostData[i] = N - i;
    }

    // Allocate device memory for input data and temporary data
    cudaMalloc((void**)&deviceData, N * sizeof(int));
    cudaMalloc((void**)&deviceTemp, N * sizeof(int));

    // Copy input data to the device
    cudaMemcpy(deviceData, hostData, N * sizeof(int), cudaMemcpyHostToDevice);

    // Sort the data using parallel merge sort
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    mergeSort<<<numBlocks, blockSize>>>(deviceData, deviceTemp, 0, N);

    // Copy the sorted result back to the host
    cudaMemcpy(hostData, deviceData, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted output
    printf("Sorted Data: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", hostData[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(deviceData);
    cudaFree(deviceTemp);

    return 0;
}