# include <stdio.h>
# include <mpi.h>
# include <math.h>

int main(int argc, char **argv){
    int x = 2;
    MPI_Init(NULL, NULL);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double P = pow(x , rank);
    printf("%d to the power of %d is %f \n" , x , rank , P);
    MPI_Finalize();
    return 0;
}