# include <stdio.h>
# include <stdlib.h>

#define N 4
#define M 4

__global__ void ones_comp(int *a, int *b, int rows, int cols){
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row > 0 && row < rows - 1 && col > 0 && col < cols - 1){
        int temp_1 = a[row * cols + cols];
        while(temp_1 > 0){
            int rem = temp_1 % 2;
            b[row * cols + cols] = b[row * cols + col] * 10 + rem;
            temp_1 = temp_1 / 2;
        }

        int temp_2 = b[row * cols + col];
        while(temp_2 > 0){
            int rem = temp_2 % 10;
            if(rem == 1){
                rem = 0;
            }
            else{
                rem = 1;
            }
            b[row * cols + col] = b[row * cols + col] * 10 + rem;
            temp_2 = temp_2 / 10;
        }
    }
    else{
        b[row * cols + col] = a[row * cols + col];
    }

}

int main(){
    int a[M][N]; 
    int b[M][N], *d_a, *d_b;

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            a[i][j] = i * N + j + 1;
        }
    }

    cudaMalloc((void **)&d_a, M * N * sizeof(int));
    cudaMalloc((void **)&d_b, M * N * sizeof(int));

    cudaMemcpy(d_a, a, M * N * sizeof(int), cudaMemcpyHostToDevice);

    ones_comp<<<M, N>>>(d_a, d_b, M, N);

    cudaMemcpy(b, d_b, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Updated Matrix : \n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            printf("%d\t", b[i][j]);
        }
        printf("\n");
    }

    free(a);
    free(b);
    cudaFree(a);
    cudaFree(b);
}