#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#include "kmeans.h"

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

int main(int argc, char **argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		fprintf(stderr, "O programa requer pelo menos 2 processos MPI.\n");
		MPI_Finalize();
		return 1;
	}

	if (rank == 0)
	{
		// Código do mestre

		// Configuração do K-means
		kmeans_config config;
		// Preencha config com os parâmetros e dados necessários...

		// Distribuição das tarefas
		int num_slaves = size - 1;
		int tasks_per_slave = 60 / num_slaves;
		int remaining_tasks = 60 % num_slaves;

		for (int i = 1; i <= num_slaves; i++)
		{
			int tasks = tasks_per_slave;

			if (remaining_tasks > 0)
			{
				tasks++;
				remaining_tasks--;
			}

			MPI_Send(&tasks, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		// Processamento adicional do mestre...

		// Recebimento dos resultados
		for (int i = 1; i <= num_slaves; i++)
		{
			kmeans_result result;
			MPI_Recv(&result, sizeof(kmeans_result), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			// Processar o resultado do escravo...
		}
	}
	else
	{
		// Código do escravo

		// Solicitar tarefas ao mestre
		int tasks;
		MPI_Recv(&tasks, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (int i = 0; i < tasks; i++)
		{
			// Gerar conjunto de pontos para o escravo
			int nptsincluster = 100000;
			int k = 6;
			int spread = 3;

			srand(rank * tasks + i);

			// Configuração dos pontos para o escravo
			kmeans_config config;
			config.k = k;
			config.num_objs = config.k * nptsincluster;
			config.max_iterations = 200;
			config.distance_method = pt_distance;
			config.centroid_method = pt_centroid;

			config.objs = calloc(config.num_objs, sizeof(point *));
			config.centers = calloc(config.k, sizeof(point *));
			config.clusters = calloc(config.num_objs, sizeof(int));

			point *pts = calloc(config.num_objs, sizeof(point));
			point *init = calloc(config.k, sizeof(point));

			for (int j = 0; j < config.k; j++)
			{
				for (int n = 0; n < nptsincluster; n++)
				{
					double u1 = 1.0 * rand() / RAND_MAX;
					double u2 = 1.0 * rand() / RAND_MAX;
					double z1 = spread * j + sqrt(-2 * log2(u1)) * cos(2 * M_PI * u2);
					double z2 = spread * j + sqrt(-2 * log2(u1)) * sin(2 * M_PI * u2);
					int index = j * nptsincluster + n;

					pts[index].x = z1;
					pts[index].y = z2;

					config.objs[index] = &(pts[index]);
				}
			}

			for (int n = 0; n < config.k; n++)
			{
				int r = lround(config.num_objs * (1.0 * rand() / RAND_MAX));
				init[n] = pts[r];
				config.centers[n] = &(init[n]);
			}

			// Executar o kmeans
			kmeans_result result = kmeans(&config);

			// Enviar o resultado para o mestre
			MPI_Send(&result, sizeof(kmeans_result), MPI_BYTE, 0, 0, MPI_COMM_WORLD);

			// Liberar memória
			free(config.objs);
			free(config.clusters);
			free(config.centers);

			free(pts);
			free(init);
		}
	}

	MPI_Finalize();
	return 0;
}
