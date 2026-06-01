from numba import cuda
import numpy as np 

@cuda.jit
def hello_kernel():
    tx = cuda.threadIdx.x
    bx= cuda.blockIdx.x
    bdim = cuda.blockDim.x

    gid = tx + bx * bdim
    gid = tx + bx * bdim

    # CUDA-safe print (no strings!)
    print("Hello from Block:",bx,"Thread:", tx, "Global Id: ",gid)

blocks = 2
threads_per_block = 4

hello_kernel[blocks, threads_per_block]()
cuda.synchronize()