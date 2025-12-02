/*
CUDA - compute average value of N numbers in range <A;B>
      VERSION WITHOUT SHARED MEMORY (64-bit sum)
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

__global__ void computeSumRaw(int *data, unsigned long long *globalSum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(globalSum, (unsigned long long)data[idx]);
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

    computeSumRaw<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, sumDevice, N);

    unsigned long long sumHostULL = 0;
    cudaMemcpy(&sumHostULL, sumDevice, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // interpretujemy jako signed (na wypadek liczb ujemnych)
    long long sumHost = (long long)sumHostULL;
    double average = (double)sumHost / (double)N;

    printf("RAW VERSION (64-bit sum)\n");
    printf("Range = [%d, %d]\n", A, B);
    printf("Sum = %lld\n", sumHost);
    printf("Average = %f\n", average);
    printf("Kernel time: %.3f ms\n", milliseconds);

    free(randomNumbers);
    cudaFree(randomNumbersDevice);
    cudaFree(sumDevice);

    return 0;
}
