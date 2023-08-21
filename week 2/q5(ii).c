#include <stdio.h>
#include <mpi.h>
int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        printf("This program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        MPI_Send("Hello", 5, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(NULL, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Send("Hello", 5, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(NULL, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Finalize();
    return 0;
}

