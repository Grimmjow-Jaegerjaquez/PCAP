#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_SENTENCE_LENGTH 1024
#define MAX_WORD_LENGTH 64

__global__ void countWordOccurrences(char* sentence, char* word, int* result, int sentenceLength, int wordLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;

    if (tid < sentenceLength) {
        int i = tid;
        int j = 0;
        while (j < wordLength && sentence[i] == word[j]) {
            i++;
            j++;
        }

        if (j == wordLength) {
            count = 1;
        }
    }

    atomicAdd(result, count);
}

int main() {
    char sentence[MAX_SENTENCE_LENGTH] = "This is a CUDA program. CUDA is a parallel computing platform.";
    char word[MAX_WORD_LENGTH] = "CUDA";

    int result = 0;

    int sentenceLength = strlen(sentence);
    int wordLength = strlen(word);

    char* d_sentence;
    char* d_word;
    int* d_result;

    cudaMalloc((void**)&d_sentence, sentenceLength * sizeof(char));
    cudaMalloc((void**)&d_word, wordLength * sizeof(char));
    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_sentence, sentence, sentenceLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wordLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (sentenceLength + blockSize - 1) / blockSize;

    countWordOccurrences<<<gridSize, blockSize>>>(d_sentence, d_word, d_result, sentenceLength, wordLength);

    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the sentence.\n", word, result);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_result);

    return 0;
}
