#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void oddEvenSort(int *localArray, int localSize, int myRank, int numProcs) {
    int *tempArray = (int *)malloc(localSize * sizeof(int));

    for (int phase = 0; phase < numProcs; phase++) {
        if (phase % 2 == 0) {
            if (myRank % 2 == 0) {
                if (myRank < numProcs - 1) {
                    MPI_Recv(tempArray, localSize, MPI_INT, myRank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int i = 0; i < localSize; i++) {
                        if (localArray[i] > tempArray[0]) {
                            int temp = localArray[i];
                            localArray[i] = tempArray[0];
                            tempArray[0] = temp;
                        }
                    }
                    MPI_Send(tempArray, localSize, MPI_INT, myRank + 1, 0, MPI_COMM_WORLD);
                }
            } else {
                MPI_Send(localArray, localSize, MPI_INT, myRank - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(localArray, localSize, MPI_INT, myRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            if (myRank % 2 == 1 && myRank < numProcs - 1) {
                MPI_Recv(tempArray, localSize, MPI_INT, myRank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < localSize; i++) {
                    if (localArray[i] > tempArray[0]) {
                        int temp = localArray[i];
                        localArray[i] = tempArray[0];
                        tempArray[0] = temp;
                    }
                }
                MPI_Send(tempArray, localSize, MPI_INT, myRank + 1, 0, MPI_COMM_WORLD);
            } else if (myRank % 2 == 0) {
                MPI_Send(localArray, localSize, MPI_INT, myRank - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(localArray, localSize, MPI_INT, myRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    free(tempArray);
}

int main(int argc, char** argv) {
    int myRank, numProcs;
    int *localArray;
    int localSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    localSize = 5; // Define the local size of each process
    localArray = (int *)malloc(localSize * sizeof(int));

    // Initialize the local array with random values
    for (int i = 0; i < localSize; i++) {
        localArray[i] = rand() % 100;
    }

    printf("Process %d local array before sorting: ", myRank);
    for (int i = 0; i < localSize; i++) {
        printf("%d ", localArray[i]);
    }
    printf("\n");

    oddEvenSort(localArray, localSize, myRank, numProcs);

    printf("Process %d local array after sorting: ", myRank);
    for (int i = 0; i < localSize; i++) {
        printf("%d ", localArray[i]);
    }
    printf("\n");

    free(localArray);

    MPI_Finalize();

    return 0;
}