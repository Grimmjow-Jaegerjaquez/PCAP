#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main(int argc, char** argv) {
    int rank, num_procs;
    int N = 5; // Number of terms (change this as needed)
    int local_sum, global_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs < 2) {
        fprintf(stderr, "This program requires at least 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (N < num_procs) {
        fprintf(stderr, "Number of terms (N) should be greater than or equal to the number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    local_sum = factorial(rank + 1);
    MPI_Scan(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == num_procs - 1) {
        printf("Process %d: Sum of factorials up to %d! = %d\n", rank, N, global_sum);
    }

    MPI_Finalize();

    return 0;
}