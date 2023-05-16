/*-------------------------------------------------------------------------
 *
 * kmeans.c
 *    Generic k-means implementation
 *
 * Copyright (c) 2016, Paul Ramsey <pramsey@cleverelephant.ca>
 *
 *------------------------------------------------------------------------*/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "kmeans.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kmeans.h"

#define MASTER_RANK 0
#define KMEANS_TAG 1

void master(kmeans_config *config)
{
	int num_procs, num_slaves;
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	num_slaves = num_procs - 1;

	// Dividir os dados entre os escravos
	int objs_per_slave = config->num_objs / num_slaves;
	int extra_objs = config->num_objs % num_slaves;
	int send_count = objs_per_slave;

	for (int slave_rank = 1; slave_rank <= num_slaves; slave_rank++)
	{
		// Ajustar o número de objetos enviados caso haja sobra
		if (slave_rank <= extra_objs)
			send_count++;

		// Enviar o número de objetos que o escravo receberá
		MPI_Send(&send_count, 1, MPI_INT, slave_rank, KMEANS_TAG, MPI_COMM_WORLD);

		// Enviar os objetos para o escravo
		MPI_Send(&(config->objs[(slave_rank - 1) * objs_per_slave]), send_count, MPI_POINTER, slave_rank, KMEANS_TAG, MPI_COMM_WORLD);
	}

	// Receber os resultados dos escravos e atualizar os centros de cluster
	for (int slave_rank = 1; slave_rank <= num_slaves; slave_rank++)
	{
		int recv_count;
		MPI_Recv(&recv_count, 1, MPI_INT, slave_rank, KMEANS_TAG, MPI_COMM_WORLD, &status);

		int *slave_clusters = malloc(sizeof(int) * recv_count);
		MPI_Recv(slave_clusters, recv_count, MPI_INT, slave_rank, KMEANS_TAG, MPI_COMM_WORLD, &status);

		// Atualizar os centros de cluster usando os resultados do escravo
		for (int i = 0; i < recv_count; i++)
		{
			int obj_index = (slave_rank - 1) * objs_per_slave + i;
			config->clusters[obj_index] = slave_clusters[i];
		}

		free(slave_clusters);
	}
}

void slave(kmeans_config *config)
{
	int num_objs, recv_count;
	MPI_Status status;

	MPI_Recv(&recv_count, 1, MPI_INT, MASTER_RANK, KMEANS_TAG, MPI_COMM_WORLD, &status);

	// Receber os objetos do mestre
	config->num_objs = recv_count;
	config->objs = malloc(sizeof(Pointer) * recv_count);
	MPI_Recv(config->objs, recv_count, MPI_POINTER, MASTER_RANK, KMEANS_TAG, MPI_COMM_WORLD, &status);

	// Executar o k-means no subconjunto de objetos
	update_r(config);

	// Enviar os resultados para o mestre
	MPI_Send(&recv_count, 1, MPI_INT, MASTER_RANK, KMEANS_TAG, MPI_COMM_WORLD);
	MPI_Send(config->clusters, recv_count, MPI_INT, MASTER_RANK, KMEANS_TAG, MPI_COMM_WORLD);

	free(config->objs);
}

kmeans_result kmeans_parallel(kmeans_config *config)
{
	int rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == MASTER_RANK)
	{
		master(config);
	}
	else
	{
		slave(config);
	}

	MPI_Finalize();
	return KMEANS_OK;
}

static void
update_means(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->k; i++)
	{
		/* Update the centroid for this cluster */
		(config->centroid_method)(config->objs, config->clusters, config->num_objs, i, config->centers[i]);
	}
}

kmeans_result
kmeans(kmeans_config *config)
{
	int iterations = 0;
	int *clusters_last;
	size_t clusters_sz = sizeof(int) * config->num_objs;

	assert(config);
	assert(config->objs);
	assert(config->num_objs);
	assert(config->distance_method);
	assert(config->centroid_method);
	assert(config->centers);
	assert(config->k);
	assert(config->clusters);
	assert(config->k <= config->num_objs);

	/* Zero out cluster numbers, just in case user forgets */
	memset(config->clusters, 0, clusters_sz);

	/* Set default max iterations if necessary */
	if (!config->max_iterations)
		config->max_iterations = KMEANS_MAX_ITERATIONS;

	/*
	 * Previous cluster state array. At this time, r doesn't mean anything
	 * but it's ok
	 */
	clusters_last = kmeans_malloc(clusters_sz);

	while (1)
	{
		/* Store the previous state of the clustering */
		memcpy(clusters_last, config->clusters, clusters_sz);

		update_r(config);
		update_means(config);
		/*
		 * if all the cluster numbers are unchanged since last time,
		 * we are at a stable solution, so we can stop here
		 */
		// if (memcmp(clusters_last, config->clusters, clusters_sz) == 0)
		// {
		// 	kmeans_free(clusters_last);
		// 	config->total_iterations = iterations;
		// 	return KMEANS_OK;
		// }

		if (iterations++ > config->max_iterations)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_EXCEEDED_MAX_ITERATIONS;
		}
	}

	kmeans_free(clusters_last);
	config->total_iterations = iterations;
	return KMEANS_ERROR;
}
