# include <stdio.h>
# include <mpi.h>

int main(int argc, char **argv){
    int rank, size, a = 10, b = 4;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        printf("Process %d : %d + %d = %d\n", rank, a, b, a + b);
    }

    else if(rank == 1){
        printf("Process %d : %d - %d = %d\n", rank, a, b, a - b);
    }

    else if(rank == 2){
        printf("Process %d : %d * %d = %d\n", rank, a, b, a*b);
    }

    else if(rank == 3){
        printf("Process %d : %d / %d = %d\n", rank, a, b, a/b);
    }

    else{
        printf("Process %d : %d mod %d = %d\n", rank, a, b, a % b);
    }

    MPI_Finalize();
    return 0;
}