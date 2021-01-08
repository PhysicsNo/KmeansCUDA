#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand_kernel.h>

#include "vec3.cuh"
#include "random.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )


void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

static float epsilon = 1.0f;

/**
Updates cluster means.
@param K - the number of clusters.
@param N - the number of objects
@param objects - the objects themselves. (for now they'll just be represented by their centers)
@param means - the means of the clusters. (should be an N-length array of zero vectors)
@param closest - index of closest cluster (mean for the nth object).
*/
__global__ void update_means(int K, int N, vec3* objects, vec3* means, int* closest, float* counts)
{
    int n = threadIdx.x;
    int k = closest[n];
    means[k] += objects[n];//objects[n].center;
    counts[k] += 1.0f;

    __syncthreads();
    means[k] /= counts[k];
}



/**
Assigns each data point to its closest centroid.
@param K - the number of clusters.
@param distances - array storing this distance to each centroid for each object.
@param closest - index of closest cluster mean for the nth object.
@return - each object's closest centroid is stored in closest. (Access objects and closest with same index).
*/
__global__ void assign_centroids(int K, float* distances, int* closest)
{
    float closest_dist = FLT_MAX;
    int n = threadIdx.x;

    for (int i = 0; i < K; i++)
    {
        if (distances[i] < closest_dist) {
            closest_dist = distances[i];
            closest[n] = i;
        }

    }
}

/**
Computes the distances to each centroid for each object, from its center.
@param objects - the objects themselves. (for now they'll just be represented by their centers)
@param means - the means of the clusters. (should be an N-length array, each entry is the center of a randomly selected object)
@param distances - array to save computed values.
*/
__global__ void dist_to_centroids(vec3* objects, vec3* means, float* distances)
{
    int n = blockIdx.x;
    int k = threadIdx.x;
    //distances[n + k] = norm3df(objects[n].center.x() - means[k].x(), objects[n].center.y() - means[k].y(), objects[n].center.z() - means[k].z());
    distances[n + k] = norm3df(objects[n].x() - means[k].x(), objects[n].y() - means[k].y(), objects[n].z() - means[k].z());
}

/**
Helper kernel to assist in computing the score for a given assignment of the kmeans algo.
@param size - number of entries.
@param objs - pointer to the object array.
@param score - the resultant values to use in the sum are saved here.
@param assignees - the centroids to which objs are assigned.
@param means - the values of centroids.
*/
__global__ void compute_scores(int size, vec3* objs, int* assignees, vec3* means, float* scores)
{
    int n = threadIdx.x;
    scores[n] = (objs[n] - means[n]).squared_length();
}

/**
Helper kernel to set all entries in an array to zero.
@param size - number of entries.
@param arr - pointer to the array.
*/
__global__ void zero_array(int size, float* arr)
{
    int i = threadIdx.x;
    arr[i] = 0.0f;
}

/**
Helper kernel to set all entries in a vector array to the zero vector.
@param size - number of entries.
@param arr - pointer to the array.
*/
__global__ void zero_vec_array(int size, vec3* arr)
{
    int i = threadIdx.x;
    arr[i] = vec3(0.0f, 0.0f, 0.0f);
}

/**
Checks if any object has changed cluster.
@param prev_assignments - previous cluster each obj was assigned to.
@param assignments - current cluster each obj was assigned to.
@return result - a booleam value indicating if any object's assigned cluster got updated.
*/
__global__ void changed_cluster(int* prev_assignments, int* assignments, bool* result)
{
    int i = threadIdx.x;
    if (prev_assignments[i] != assignments[i])
        *result = true;
}

/**
Updates assigned cluster array.
@param prev_assignments - previous cluster each obj was assigned to.
@param assignments - current cluster each obj was assigned to.
*/
__global__ void update_assignments(int* prev_assignments, int* assignments)
{
    int i = threadIdx.x;
    prev_assignments[i] = assignments[i];
}

/**
Performs k-means clustering to build a level of the bvh.
@param N - the number of objects.
@param objects - the objects themselves.
@param means - the means of the clusters.
@param prev_score - the score of the previous clustering assignment.
@param assignments - stores the index into the means array for each object, entry at the index is the closest centroid.
*/
void cluster(int N, int K, vec3* objects, vec3* means, int* assignments)
{
    //1. Memory setup
    //float* scores;
    float* d_distances;
    int* d_new_assignments;
    bool cluster_updated = false;
    
    //checkCudaErrors(cudaMallocManaged((void**)&scores, N * sizeof(*scores)));
    checkCudaErrors(cudaMalloc((void**)&d_distances, N * K * sizeof(*d_distances)));
    checkCudaErrors(cudaMalloc((void**)&d_new_assignments, N * sizeof(*d_new_assignments)));
    
    //2. Get distances to centroids for each obj
    dist_to_centroids<<<N, K>>>(objects, means, d_distances);
    checkCudaErrors(cudaGetLastError());

    //3. Get the closest centroid to each obj, and by doing so, "assign" each object to its closest centroid.
    assign_centroids<<<1, N>>>(K, d_distances, d_new_assignments);
    checkCudaErrors(cudaGetLastError());

    //4. Check for convergence (test this approach vs. setting just a max# itrs vs checking cluster changed)
    /*
    compute_scores<<<1, N >>>(N, objects, assignments, means, scores);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    //TODO Can def do this in parallel
    float score = 0;
    for (int i = 0; i < N; i++)
        score += scores[i];

    checkCudaErrors(cudaFree(d_distances));
    checkCudaErrors(cudaFree(scores));

    if (prev_score - score < epsilon)
        return;
    */

    changed_cluster<<<1, N>>>(assignments, d_new_assignments, &cluster_updated);
    checkCudaErrors(cudaGetLastError());
    
    update_assignments<<<1, N>>>(assignments, d_new_assignments);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_distances));
    checkCudaErrors(cudaFree(d_new_assignments));

    if (!cluster_updated)
        return;

    //5. Update cluster centers if termination condition unmet
    float* d_counts;
    checkCudaErrors(cudaMalloc((void**)&d_counts, N * sizeof(*d_counts)));
    
    zero_vec_array<<<1, N >>>(N, means);
    checkCudaErrors(cudaGetLastError());

    zero_array<<<1, N>>>(N, d_counts);
    checkCudaErrors(cudaGetLastError());

    update_means<<<1, N>>>(K, N, objects, means, assignments, d_counts);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    //6. Re-run the algo till we get convergence
    checkCudaErrors(cudaFree(d_counts));
    cluster(N, K, objects, means, assignments);
}

//May or may not use it
__global__ void setup_rand_seq_kernel(curandState* state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

/**
Initializes the means array by setting the entries as the center of a randomly selected object.
@param N - the number of objects from which to choose centroid values.
@param objects - the objects.
@param means - the means whose values are to be set.
@param rand_state - for random number generation.
*/
__global__ void init_means(int N, vec3* objects, vec3* means, curandState* rand_state)
{
    int k = threadIdx.x;
    //TODO: TEST! The goal here is to randomly, in each thread, a value from a uniform distribution of [0, N]
    curandState local_rand_state = rand_state[k];
    int rand_index = curand(&local_rand_state) % N;
    //int rand_index = (int)(N * curand_uniform(&local_rand_state)); An alt way of doing things
    means[k] = objects[rand_index];
}

int main()
{
    /*
    Open issues:
    -random initializations to get an optimal clustering? <- TODO: experiment
    -convergence evaluation metrics, figure out an efficient way of checking no object's have changed assignment <- TODO: experiment
    -handling empty clusters <- TODO: implement (this will be done during node construction)
    -then, of course, loading in datasets and testing if the algos work! <- TODO: implement
    */

    

    //1. TODO initialization
    int N = 100, K = 6;
    vec3* d_objects;//TODO: read 'em in
    vec3* d_means;
    int* d_assignments;
    curandState* d_rand_state;
    float score = FLT_MAX;
    
    checkCudaErrors(cudaMalloc((void**)&d_objects, N * sizeof(*d_objects)));
    checkCudaErrors(cudaMalloc((void**)&d_means, K * sizeof(*d_means)));
    checkCudaErrors(cudaMalloc((void**)&d_assignments, N * sizeof(*d_assignments)));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, N * sizeof(*d_rand_state)));

    setup_rand_seq_kernel<<<1, N>>>(d_rand_state);
    checkCudaErrors(cudaGetLastError());

    init_means<<<1, K>>>(N, d_objects, d_means, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cluster(N, K, d_objects, d_means, score, d_assignments);
    
    /*
    Then you build a new bvh node, with center = the mean and a list of children = all objects whose mean = this node's center.
    The way to streamline that is just by using the index:
    1. Build K bvh nodes with appropriately set centers.
    2. Launch a kernel with N threads, params are: objects, assignments, (new) nodes
    3. k = assignments[n]; nodes[k].children.add(objects[n]);
    4. Probably do another Kernel to get the bounds/check for empties!
    */
    checkCudaErrors(cudaFree(d_objects));
    checkCudaErrors(cudaFree(d_means));
    checkCudaErrors(cudaFree(d_assignments));
    return 0;
}