#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_STRING_LENGTH 1024
#define MAX_WORD_LENGTH 64

__device__ void reverseWord(char* word, int start, int end) {
    while (start < end) {
        char temp = word[start];
        word[start] = word[end];
        word[end] = temp;
        start++;
        end--;
    }
}

__global__ void reverseWords(char* input, int* wordLengths, int numWords) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numWords) {
        int start = tid == 0 ? 0 : wordLengths[tid - 1] + 1;
        int end = wordLengths[tid];
        reverseWord(input, start, end);
    }
}

int main() {
    char input[MAX_STRING_LENGTH] = "Hello CUDA World";
    int numWords = 0;
    int wordLengths[MAX_STRING_LENGTH];
    char* d_input;
    int* d_wordLengths;

    // Tokenize the input string and store word lengths
    char* token = strtok(input, " ");
    while (token != NULL) {
        int length = strlen(token);
        wordLengths[numWords] = length - 1; // Exclude the null terminator
        numWords++;
        token = strtok(NULL, " ");
    }

    cudaMalloc((void**)&d_input, MAX_STRING_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_wordLengths, numWords * sizeof(int));

    cudaMemcpy(d_input, input, MAX_STRING_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wordLengths, wordLengths, numWords * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numWords + blockSize - 1) / blockSize;

    reverseWords<<<gridSize, blockSize>>>(d_input, d_wordLengths, numWords);

    cudaMemcpy(input, d_input, MAX_STRING_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Original string: Hello CUDA World\n");
    printf("Reversed string: %s\n", input);

    cudaFree(d_input);
    cudaFree(d_wordLengths);

    return 0;
}