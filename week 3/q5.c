#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void parallelSelectionSort(int *arr, int size) {
    for (int i = 0; i < size - 1; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < size; ++j) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            swap(&arr[min_idx], &arr[i]);
        }
    }
}

void mergeSortedArrays(int *arr, int size, int *subarray, int subarray_size) {
    int *merged = (int *)malloc((size + subarray_size) * sizeof(int));

    int i = 0, j = 0, k = 0;
    while (i < size && j < subarray_size) {
        if (arr[i] < subarray[j]) {
            merged[k++] = arr[i++];
        } else {
            merged[k++] = subarray[j++];
        }
    }

    while (i < size) {
        merged[k++] = arr[i++];
    }

    while (j < subarray_size) {
        merged[k++] = subarray[j++];
    }

    for (int idx = 0; idx < size + subarray_size; ++idx) {
        arr[idx] = merged[idx];
    }

    free(merged);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int array_size;
    if (rank == 0) {
        printf("Enter the size of the array: ");
        scanf("%d", &array_size);
    }

    MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = array_size / size;
    int *local_data = (int *)malloc(local_size * sizeof(int));

    MPI_Scatter(NULL, 0, MPI_INT, local_data, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    parallelSelectionSort(local_data, local_size);

    int *sorted_array = NULL;
    if (rank == 0) {
        sorted_array = (int *)malloc(array_size * sizeof(int));
    }

    MPI_Gather(local_data, local_size, MPI_INT, sorted_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        parallelSelectionSort(sorted_array, array_size);

        printf("Sorted array:\n");
        for (int i = 0; i < array_size; ++i) {
            printf("%d ", sorted_array[i]);
        }
        printf("\n");

        free(sorted_array);
    }

    free(local_data);

    MPI_Finalize();

    return 0;
}