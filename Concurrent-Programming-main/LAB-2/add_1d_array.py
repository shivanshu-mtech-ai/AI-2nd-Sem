from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(a, b, c):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    gid = tx + bx * bdim 

    if gid < a.size:
        c[gid] = a[gid] + b[gid]

a1 = np.array([1,2,3,4,5,6],dtype=np.int32)
a2 = np.array([10,20,30,40,50,60],dtype=np.int32)
a3 = np.zeros(6,dtype=np.int32)

d_a1 = cuda.to_device(a1)
d_a2 = cuda.to_device(a2)
d_a3 = cuda.to_device(a3)

blocks=2
threads_per_block = 4

add_kernel[blocks, threads_per_block](d_a1, d_a2, d_a3)
cuda.synchronize()

result = d_a3.copy_to_host()
print(result)

