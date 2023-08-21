# include <stdio.h>
# include <mpi.h>
# include <string.h>

int main(int argc, char **argv){
    int rank, size, i, j, m;
    float avg = 0;
    char b[100], str1[100], str2[100], c1[100], c2[100], concatted[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        printf("Enter string 1: ");
		scanf("%s", str1);
		
		printf("Enter string 2: ");
		scanf("%s", str2);

		m = strlen(str1) / size;
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Scatter(str1, m, MPI_CHAR, c1, m, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Scatter(str2, m, MPI_CHAR, c2, m, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	for(int t = 0; t <= 2 * m; t += 2){
        concatted[t] = c1[t/2];
		concatted[t+1] = c2[t/2];
    }
    concatted[2*m] = '\0';
    MPI_Gather(concatted, 2*m, MPI_CHAR, b, 2*m, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	if (rank == 0) 
	{
		b[m*size*2] = '\0';
		printf("Concatted: %s", b);
	}

	MPI_Finalize();
    return 0;
}