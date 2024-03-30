#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tuple>

__device__ float deviceMultiply(float a, float b)
{
    return a * b;
}

__global__ void vectorMult(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = deviceMultiply(A[i], B[i]);
    }
}

__host__ std::tuple<float *, float *, float *> allocateHostMemory(int numElements)
{
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    return std::make_tuple(h_A, h_B, h_C);
}

__host__ void executeKernel(float *d_A, float *d_B, float *d_C, int numElements)
{
    // Launch the Vector Multiply CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorMult kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    int numElements = 50000;
    printf("[Vector multiplication of %d elements]\n", numElements);

    float *h_A, *h_B, *h_C;
    std::tie(h_A, h_B, h_C) = allocateHostMemory(numElements);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, numElements * sizeof(float));
    cudaMalloc(&d_B, numElements * sizeof(float));
    cudaMalloc(&d_C, numElements * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Execute the kernel
    executeKernel(d_A, d_B, d_C, numElements);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs((h_A[i] * h_B[i]) - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
