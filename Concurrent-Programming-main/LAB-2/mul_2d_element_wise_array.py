from numba import cuda
import numpy as np

@cuda.jit
def add_matrices(a, b, c, n, m):
    i, j = cuda.grid(2)
    if i < n and j < m:
        c[i, j] = a[i, j] * b[i, j]

def main():
    n = 6
    m = 6
    a = np.arange(n * m, dtype=np.int32).reshape((n, m))
    b = np.arange(n * m, dtype=np.int32).reshape((n, m))
    c = np.zeros((n, m), dtype=np.int32)
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)
    threads_per_block = (6, 6)
    blocks_per_grid = ((n + threads_per_block[0] - 1),
                       (m + threads_per_block[1] - 1))
    add_matrices[blocks_per_grid, threads_per_block](d_a, d_b, d_c, n, m)
    cuda.synchronize()
    c = d_c.copy_to_host()
    print("First 10 rows and columns of the result:")
    print(c[:10, :10])

if __name__ == "__main__":
    main()
