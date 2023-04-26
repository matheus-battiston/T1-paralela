#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include "kmeans.h"

int NUMERO_K = 12;
int NPTSINCLUSTER = 100000;
int NUMERO_CORES = 1;

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
	omp_set_num_threads(NUMERO_CORES);

	kmeans_config config;
	kmeans_result result;
	int i, j;
	int spread = 3;
	point *pts;
	point *init;
	int print_results = 0;
	unsigned long start;

	int nptsincluster = NPTSINCLUSTER;
	int k = NUMERO_K;

	srand(time(NULL));

	/* Constants */
	config.k = k;
	config.num_objs = config.k * nptsincluster;
	config.max_iterations = 200;
	config.distance_method = pt_distance;
	config.centroid_method = pt_centroid;

	/* Inputs for K-means */
	config.objs = calloc(config.num_objs, sizeof(Pointer));
	config.centers = calloc(config.k, sizeof(Pointer));
	config.clusters = calloc(config.num_objs, sizeof(int));

	/* Storage for raw data */
	pts = calloc(config.num_objs, sizeof(point));
	init = calloc(config.k, sizeof(point));

	/* Create test data! */
	/* Populate with K gaussian clusters of data */
	for (j = 0; j < config.k; j++)
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
			config.objs[n] = &(pts[n]);
		}
	}

	/* Populate the initial means vector with random start points */
	for (i = 0; i < config.k; i++)
	{
		int r = lround(config.num_objs * (1.0 * rand() / RAND_MAX));
		/* Populate raw data */
		init[i] = pts[r];
		/* Pointers to raw data */
		config.centers[i] = &(init[i]);

		if (print_results)
			printf("center[%d]\t%g\t%g\n", i, init[i].x, init[i].y);
	}

	/* run k-means! */
	start = time(NULL);
	starttime = omp_get_wtime();

	result = kmeans(&config);

	stoptime = omp_get_wtime();

	printf("\nTempo de execucao: %3.2f segundos\n\n", stoptime - starttime);

	// printf("\n");
	// printf("Iteration count: %d\n", config.total_iterations);
	// printf("     Time taken: %ld seconds\n", (time(NULL) - start));
	// printf(" Iterations/sec: %.3g\n", (1.0 * config.total_iterations) / (time(NULL) - start));
	// printf("\n");

	// /* print results */
	// if (print_results)
	// {
	// 	for (i = 0; i < config.num_objs; i++)
	// 	{
	// 		point *pt = (point *)(config.objs[i]);

	// 		if (config.objs[i])
	// 			printf("%g\t%g\t%d\n", pt->x, pt->y, config.clusters[i]);
	// 		else
	// 			printf("N\tN\t%d\n", config.clusters[i]);
	// 	}
	// }

	free(config.objs);
	free(config.clusters);
	free(config.centers);

	free(init);
	free(pts);
}
