#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
 
int main(int argc, char *argv[]){
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {	
        int *arr = (int *)malloc(size * sizeof(int));
        for(int i=0;i<size;i++)
            scanf("%d", &arr[i]);

        for (int i = 1; i < size; i++) {
            MPI_Ssend(arr + i, sizeof(int), MPI_INT, i, 100 + i, MPI_COMM_WORLD);
            printf("Process %d sent %d to Process %d.\n", rank, arr[i], i);
        }
    }
    else {
        int num;
        MPI_Recv(&num, sizeof(int), MPI_INT, 0, 100 + rank, MPI_COMM_WORLD, &status);
        if (rank % 2 == 0)
            num = num * num;
        else
            num = num * num * num;
    
        printf("Process %d value: %d\n", rank, num);
 
    }
    MPI_Finalize(); 
}