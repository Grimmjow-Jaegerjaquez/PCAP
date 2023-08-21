#include <stdio.h>
#include <mpi.h>

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

int main(int argc, char** argv) {
    int rank, num_procs;
    int num_rectangles = 1000000;
    double a = 0.0, b = 1.0;
    double local_sum = 0.0, global_sum = 0.0;
    double width = (b - a) / num_rectangles;
    double local_result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int local_n = num_rectangles / num_procs;
    int start = rank * local_n;
    int end = start + local_n;

    for (int i = start; i < end; ++i) {
        double x = a + i * width;
        local_sum += f(x) + f(x + width);
    }
    local_result = local_sum * width / 2.0;

    MPI_Reduce(&local_result, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Approximation of π: %.15f\n", global_sum);
    }

    MPI_Finalize();

    return 0;
}
