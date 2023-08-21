#include <stdio.h>
#include <mpi.h>
int main(int argc, char** argv) {
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        printf("This program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        int data = 42;
        MPI_Ssend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
    } else if (rank == 1) {
        int data = 0;
        MPI_Ssend(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }--
    MPI_Finalize();
    return 0;
}