#include <stdio.h>
#include <cuda.h>

__global__ void hellokernel()
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdim = blockDim.x;
    int gid = tx + bx * bdim;
    printf("Hello from Block: %d Thread: %d Global Id: %d\n", bx, tx, gid);
}

int main()
{
    int blocks = 2;
    int threads_per_block = 4;

    hellokernel<<<blocks, threads_per_block>>>();
    cudaDeviceSynchronize();

    return 0;
}
