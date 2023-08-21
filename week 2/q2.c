# include <stdio.h>
# include <mpi.h>
# include <stdlib.h>

# define SIZE sizeof(int)

int main(int argc, char **argv){
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *num = (int *)malloc(SIZE);
    
    int i;
    
    if(rank == 0){
        *num = rand() % 10 + 1;
        for(i = 1; i < size; i++){
            printf("Process %d sends to Process %d: %d\n", rank, i, *num);
			// Send to the process with ID = i
			MPI_Send(num, SIZE, MPI_INT, i, 100 + i, MPI_COMM_WORLD);
        }
    }
    else{
        // Revc from the process with ID = 0
        MPI_Recv(num, SIZE, MPI_INT, 0, 100 + rank, MPI_COMM_WORLD, &status);
        printf("Process %d received : %d\n", rank, *num);
    }

    MPI_Finalize();
    return 0;
}