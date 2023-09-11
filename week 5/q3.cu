#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 4
#define INPUT_SIZE 12
#define MASK_WIDTH 5
__constant__ float M[MASK_WIDTH];

__global__ void conv(float *N, float *M, float *P, int Width){
	int i = blockIdx.x*blockDim.x+ threadIdx.x;
	float Pvalue = 0;
	int n_start_point = i-(MASK_WIDTH/2);
	for(int j =0; j<MASK_WIDTH;j++){
		if(n_start_point+j >=0 && n_start_point+j < Width){
			Pvalue+= N[n_start_point+j]*M[j];
		}
	}
	P[i]=Pvalue;
}

int main(){

	float *d_N = 0;
	float *d_P = 0;
    float *d_M = 0;
    float h_M[MASK_WIDTH]={2,3,4,3,2};
	cudaMalloc(&d_N,INPUT_SIZE*sizeof(float));
	cudaMalloc(&d_P,INPUT_SIZE*sizeof(float));
    cudaMalloc(&d_M,MASK_WIDTH*sizeof(float));
	float *h_N = (float*)malloc(INPUT_SIZE*sizeof(float));
	float *h_P = (float*)malloc(INPUT_SIZE*sizeof(float));
	//float *h_M = (float*)malloc(MASK_WIDTH*sizeof(float));

	for(int i=0;i<INPUT_SIZE;++i){
		h_N[i]=(float)1;
	}
    
	cudaMemcpy(d_N,h_N,INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_P,h_P,INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_M,h_M,MASK_WIDTH*sizeof(float),cudaMemcpyHostToDevice);
	
    conv<<<(INPUT_SIZE+TILE_SIZE-1)/TILE_SIZE,TILE_SIZE >>>(d_N,d_M,d_P, MASK_WIDTH);
	cudaMemcpy(h_P,d_P,INPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
	for(int i=0; i<INPUT_SIZE;++i){
		printf("%f\n", h_P[i]);
	}
	cudaFree(d_N);
	cudaFree(d_P);
	cudaFree(M);
	free(h_N);
	free(h_P);
	//free(h_M);
    return 0;
}