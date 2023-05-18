#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "kmeans.h"
#include "mpi.h"

int NUMERO_K = 12;
int NPTSINCLUSTER = 200000;
int NUM_CONJUNTOS = 60;

typedef struct point
{
	double x;
	double y;
} point;

static double pt_distance(const Pointer a, const Pointer b)
{
	point *pa = (point *)a;
	point *pb = (point *)b;

	double dx = (pa->x - pb->x);
	double dy = (pa->y - pb->y);

	return dx * dx + dy * dy;
}

static void pt_centroid(const Pointer *objs, const int *clusters, size_t num_objs, int cluster, Pointer centroid)
{
	int i;
	int num_cluster = 0;
	point sum;
	point **pts = (point **)objs;
	point *center = (point *)centroid;

	sum.x = sum.y = 0.0;

	if (num_objs <= 0)
		return;

	for (i = 0; i < num_objs; i++)
	{
		/* Only process objects of interest */
		if (clusters[i] != cluster)
			continue;

		sum.x += pts[i]->x;
		sum.y += pts[i]->y;
		num_cluster++;
	}
	if (num_cluster)
	{
		sum.x /= num_cluster;
		sum.y /= num_cluster;
		*center = sum;
	}
	return;
}

int main(int nargs, char **args)
{
	double starttime, stoptime;
	int nptsincluster = NPTSINCLUSTER;
	int k = NUMERO_K;

	kmeans_config config[NUM_CONJUNTOS];
	kmeans_result result[NUM_CONJUNTOS];

	for (int conjunto = 0; conjunto < NUM_CONJUNTOS; conjunto++)
	{
		int i, j;
		int spread = 3;
		point *pts;
		point *init;
		int print_results = 0;
		unsigned long start;
		srand(time(NULL));

		/* Constants */
		config[conjunto].k = k;
		config[conjunto].num_objs = config[conjunto].k * nptsincluster;
		config[conjunto].max_iterations = 200;
		config[conjunto].distance_method = pt_distance;
		config[conjunto].centroid_method = pt_centroid;

		/* Inputs for K-means */
		config[conjunto].objs = calloc(config[conjunto].num_objs, sizeof(Pointer));
		config[conjunto].centers = calloc(config[conjunto].k, sizeof(Pointer));
		config[conjunto].clusters = calloc(config[conjunto].num_objs, sizeof(int));

		/* Storage for raw data */
		pts = calloc(config[conjunto].num_objs, sizeof(point));
		init = calloc(config[conjunto].k, sizeof(point));

		/* Create test data! */
		/* Populate with K gaussian clusters of data */
		for (j = 0; j < config[conjunto].k; j++)
		{
			for (i = 0; i < nptsincluster; i++)
			{
				double u1 = 1.0 * i;
				double u2 = 1.0 * i;
				double z1 = spread * j + sqrt(-2 * log2(u1)) * cos(2 * M_PI * u2);
				double z2 = spread * j + sqrt(-2 * log2(u1)) * sin(2 * M_PI * u2);
				int n = j * nptsincluster + i;

				/* Populate raw data */
				pts[n].x = z1;
				pts[n].y = z2;

				/* Pointer to raw data */
				config[conjunto].objs[n] = &(pts[n]);
			}
		}

		/* Populate the initial means vector with random start points */
		for (i = 0; i < config[conjunto].k; i++)
		{
			int r = lround(config[conjunto].num_objs * (1.0 * rand() / RAND_MAX));
			/* Populate raw data */
			init[i] = pts[r];
			/* Pointers to raw data */
			config[conjunto].centers[i] = &(init[i]);

			if (print_results)
				printf("center[%d]\t%g\t%g\n", i, init[i].x, init[i].y);
		}
	}

	/* run k-means! */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	if (num_procs < 2)
	{
		fprintf(stderr, "O número de processos deve ser pelo menos 2.\n");
		MPI_Finalize();
		return 1;
	}

	srand(time(NULL) + my_rank);

	if (my_rank == 0)
	{
		// Processo mestre

		// Distribuir os conjuntos de pontos entre os processos escravos
		int conjuntos_por_escravo = NUM_CONJUNTOS / (num_procs - 1);
		int conjuntos_restantes = NUM_CONJUNTOS % (num_procs - 1);

		for (int escravo = 1; escravo < num_procs; escravo++)
		{
			int conjuntos_para_enviar = conjuntos_por_escravo;
			if (escravo <= conjuntos_restantes)
			{
				conjuntos_para_enviar++;
			}

			MPI_Send(&conjuntos_para_enviar, 1, MPI_INT, escravo, 0, MPI_COMM_WORLD);

			for (int conjunto = 0; conjunto < conjuntos_para_enviar; conjunto++)
			{
				kmeans_config config;
				// Configurar config para o conjunto atual
				// ...

				// Enviar config para o escravo
				MPI_Send(&config, sizeof(kmeans_config), MPI_BYTE, escravo, conjunto + 1, MPI_COMM_WORLD);
			}
		}

		// Receber resultados dos escravos
		kmeans_result results[NUM_CONJUNTOS];
		for (int conjunto = 0; conjunto < NUM_CONJUNTOS; conjunto++)
		{
			int escravo = conjunto % (num_procs - 1) + 1;
			MPI_Recv(&results[conjunto], sizeof(kmeans_result), MPI_BYTE, escravo, conjunto + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// Processar resultados recebidos
		// ...

		// Imprimir resultados
		// ...
	}
	else
	{
		// Processo escravo

		// Receber número de conjuntos de pontos a serem processados
		int conjuntos_a_processar;
		MPI_Recv(&conjuntos_a_processar, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (int conjunto = 0; conjunto < conjuntos_a_processar; conjunto++)
		{
			kmeans_config config;
			// Receber config do mestre
			MPI_Recv(&config, sizeof(kmeans_config), MPI_BYTE, 0, conjunto + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Realizar o processamento do conjunto de pontos
			kmeans_result result = kmeans(&config);

			// Enviar resultado para o mestre
			MPI_Send(&result, sizeof(kmeans_result), MPI_BYTE, 0, conjunto + 1, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();
	return 0;
}