#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M;
    double *data = NULL, *local_data = NULL, local_average = 0.0, *all_averages = NULL, total_average = 0.0;

    if (rank == 0) {
        printf("Enter the value of M: ");
        scanf("%d", &M);

        data = (double *)malloc(M * size * sizeof(double));
        for (int i = 0; i < M * size; ++i) {
            printf("Enter element %d: ", i + 1);
            scanf("%lf", &data[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local data
    local_data = (double *)malloc(M * sizeof(double));

    // Scatter data from root to all processes
    MPI_Scatter(data, M, MPI_DOUBLE, local_data, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate local average
    for (int i = 0; i < M; ++i) {
        local_average += local_data[i];
    }
    local_average /= M;

    // Gather local averages at root
    if (rank == 0) {
        all_averages = (double *)malloc(size * sizeof(double));
    }

    printf("Process [%d] average : %.2f\n", rank, local_average);
    MPI_Gather(&local_average, 1, MPI_DOUBLE, all_averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate total average at root
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            total_average += all_averages[i];
        }
        total_average /= size;

        printf("Total average: %lf\n", total_average);

        free(data);
        free(all_averages);
    }

    free(local_data);

    MPI_Finalize();
    return 0;
}