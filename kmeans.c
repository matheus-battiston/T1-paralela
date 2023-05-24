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

#define TAG_WORK_REQUEST 1
#define TAG_WORK_RESPONSE 2
#define TAG_TERMINATE 3

#ifdef KMEANS_THREADED
#include <pthread.h>
#endif

typedef struct
{
	int id;
	kmeans_config *config;
	int total_iterations;
} slave_args;

void slave_kmeans(kmeans_config *config)
{
	while (1)
	{
		memcpy(clusters_last, config->clusters, clusters_sz);

		update_r(config);
		update_means(config);

		if (memcmp(clusters_last, config->clusters, clusters_sz) == 0)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_OK;
		}

		if (iterations++ > config->max_iterations)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_EXCEEDED_MAX_ITERATIONS;
		}
	}
}

kmeans_result kmeans(kmeans_config *config)
{
	assert(config);
	assert(config->objs);
	assert(config->num_objs);
	assert(config->distance_method);
	assert(config->centroid_method);
	assert(config->centers);
	assert(config->k);
	assert(config->clusters);
	assert(config->k <= config->num_objs);

	int num_procs, my_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank == 0)
	{
		// Mestre
		size_t clusters_sz = sizeof(int) * config->num_objs;
		memset(config->clusters, 0, clusters_sz);

		if (!config->max_iterations)
			config->max_iterations = KMEANS_MAX_ITERATIONS;

		int num_slaves = num_procs - 1;
		int num_objs_per_slave = config->num_objs / num_slaves;
		int remaining_objs = config->num_objs % num_slaves;
		int obj_index = 0;

		// Enviar trabalho para os escravos
		for (int i = 1; i <= num_slaves; i++)
		{
			int num_objs_slave = num_objs_per_slave;
			if (i <= remaining_objs)
				num_objs_slave++;

			MPI_Send(&num_objs_slave, 1, MPI_INT, i, TAG_WORK_REQUEST, MPI_COMM_WORLD);
			MPI_Send(&(config->objs[obj_index]), num_objs_slave, MPI_INT, i, TAG_WORK_REQUEST, MPI_COMM_WORLD);

			obj_index += num_objs_slave;
		}

		// Coletar resultados dos escravos
		int total_iterations = 0;
		for (int i = 1; i <= num_slaves; i++)
		{
			int iterations;
			MPI_Recv(&iterations, 1, MPI_INT, i, TAG_WORK_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			total_iterations += iterations;
		}

		config->total_iterations = total_iterations;

		// Terminar os escravos
		for (int i = 1; i <= num_slaves; i++)
		{
			MPI_Send(NULL, 0, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD);
		}
	}
	else
	{
		// Escravo
		while (1)
		{
			int num_objs_slave;
			MPI_Recv(&num_objs_slave, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			if (num_objs_slave == 0)
				break; // Terminar

			config->num_objs = num_objs_slave;
			MPI_Recv(config->objs, num_objs_slave, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			slave_kmeans(config);

			MPI_Send(&(config->total_iterations), 1, MPI_INT, 0, TAG_WORK_RESPONSE, MPI_COMM_WORLD);
		}
	}
}

static void
update_r(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->num_objs; i++)
	{
		double distance, curr_distance;
		int cluster, curr_cluster;
		Pointer obj;

		assert(config->objs != NULL);
		assert(config->num_objs > 0);
		assert(config->centers);
		assert(config->clusters);

		obj = config->objs[i];
		if (!obj)
		{
			config->clusters[i] = KMEANS_NULL_CLUSTER;
			continue;
		}

		/* Initialize with distance to first cluster */
		curr_distance = (config->distance_method)(obj, config->centers[0]);
		curr_cluster = 0;

		/* Check all other cluster centers and find the nearest */
		for (cluster = 1; cluster < config->k; cluster++)
		{
			distance = (config->distance_method)(obj, config->centers[cluster]);
			if (distance < curr_distance)
			{
				curr_distance = distance;
				curr_cluster = cluster;
			}
		}

		/* Store the nearest cluster this object is in */
		config->clusters[i] = curr_cluster;
	}
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
