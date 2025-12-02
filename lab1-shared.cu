/*
CUDA - compute average value of N numbers in range <A;B>
      VERSION WITH SHARED MEMORY (64-bit sum)
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

__global__ void computeSumSharedMemory(int *data, unsigned long long *globalSum, int N) {
    extern __shared__ unsigned long long sharedSum[];

    int tid       = threadIdx.x;
    int globalId  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride    = blockDim.x * gridDim.x;

    unsigned long long localSum = 0;
    for (int i = globalId; i < N; i += stride) {
        localSum += (unsigned long long)data[i];
    }

    sharedSum[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(globalSum, sharedSum[0]);
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }
}

int main() {

    int threadsinblock = 1024;
    int blocksingrid;

    int N;
    int A = 0;
    int B = 100;

    cudaEvent_t start, stop;
    float milliseconds = 0;

    printf("Enter number of elements:\n");
    scanf("%d", &N);

    if (N <= 0) {
        printf("N must be > 0\n");
        return 1;
    }

    int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (!randomNumbers) {
        printf("Memory allocation failed\n");
        return 1;
    }
    generateRandomNumbers(randomNumbers, N, A, B);

    blocksingrid = (int)ceil((double)N / threadsinblock);
    printf("The kernel will run with: %d blocks\n", blocksingrid);

    int *randomNumbersDevice;
    unsigned long long *sumDevice;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void**)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void**)&sumDevice, sizeof(unsigned long long));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(sumDevice, 0, sizeof(unsigned long long));

    int sharedMemSize = threadsinblock * sizeof(unsigned long long);

    computeSumSharedMemory<<<blocksingrid, threadsinblock, sharedMemSize>>>(randomNumbersDevice, sumDevice, N);

    unsigned long long sumHostULL = 0;
    cudaMemcpy(&sumHostULL, sumDevice, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    long long sumHost = (long long)sumHostULL;
    double average = (double)sumHost / (double)N;

    printf("SHARED VERSION (64-bit sum)\n");
    printf("Range = [%d, %d]\n", A, B);
    printf("Sum = %lld\n", sumHost);
    printf("Average = %f\n", average);
    printf("Kernel time: %.3f ms\n", milliseconds);

    free(randomNumbers);
    cudaFree(randomNumbersDevice);
    cudaFree(sumDevice);

    return 0;
}
