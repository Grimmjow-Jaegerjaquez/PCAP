#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 3

int main(int argc, char** argv) {
    int rank, num_procs;
    int search_element, local_count = 0, global_count = 0;
    int matrix[MATRIX_SIZE][MATRIX_SIZE];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        printf("Enter the 3x3 matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                scanf("%d", &matrix[i][j]);
            }
        }
        
        printf("Enter the element to be searched: ");
        scanf("%d", &search_element);
    }

    MPI_Bcast(&search_element, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    int local_start = (MATRIX_SIZE * MATRIX_SIZE / num_procs) * rank;
    int local_end = local_start + (MATRIX_SIZE * MATRIX_SIZE / num_procs);
    
    for (int i = local_start; i < local_end; ++i) {
        int row = i / MATRIX_SIZE;
        int col = i % MATRIX_SIZE;
        if (matrix[row][col] == search_element) {
            local_count++;
        }
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Number of occurrences of %d in the matrix: %d\n", search_element, global_count);
    }

    MPI_Finalize();

    return 0;
}