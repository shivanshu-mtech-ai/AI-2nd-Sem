import math
import os
import time
import numpy as np
from numba import cuda
from PIL import Image

@cuda.jit
def rgb_to_gray_gpu(rgb_img, gray_img):
    x, y = cuda.grid(2)

    if x < rgb_img.shape[0] and y < rgb_img.shape[1]:
        r = rgb_img[x, y, 0]
        g = rgb_img[x, y, 1]
        b = rgb_img[x, y, 2]

        gray_img[x, y] = 0.299*r + 0.587*g + 0.114*b


def rgb_to_gray_cpu(rgb):
    height, width = rgb.shape[:2]
    gray = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            r, g, b = rgb[i, j]
            gray[i, j] = 0.299*r + 0.587*g + 0.114*b

    return gray


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_image = os.path.join(BASE_DIR, "images", "input_1.jpg")
    output_cpu = os.path.join(BASE_DIR, "images", "gray_cpu.jpg")
    output_gpu = os.path.join(BASE_DIR, "images", "gray_gpu.jpg")

    img = Image.open(input_image).convert("RGB")
    rgb = np.array(img).astype(np.float32)

    height, width = rgb.shape[:2]


    start_cpu = time.perf_counter()

    gray_cpu = rgb_to_gray_cpu(rgb)

    cpu_time = time.perf_counter() - start_cpu

    Image.fromarray(np.clip(gray_cpu, 0, 255).astype(np.uint8)).save(output_cpu)


    gray_gpu = np.zeros((height, width), dtype=np.float32)

    start_gpu = time.perf_counter()

    d_rgb = cuda.to_device(rgb)
    d_gray = cuda.to_device(gray_gpu)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        math.ceil(height / 16),
        math.ceil(width / 16),
    )

    rgb_to_gray_gpu[blocks_per_grid, threads_per_block](d_rgb, d_gray)

    cuda.synchronize()  

    gray_result = d_gray.copy_to_host()

    gpu_time = time.perf_counter() - start_gpu

    Image.fromarray(np.clip(gray_result, 0, 255).astype(np.uint8)).save(output_gpu)

    print("\n===== CPU vs GPU Comparison =====")
    print(f"CPU Time : {cpu_time:.6f} seconds")
    print(f"GPU Time : {gpu_time:.6f} seconds")
    print(f"Speedup  : {cpu_time/gpu_time:.2f}x")

    print("\nImages saved:")
    print("CPU Output -> gray_cpu.jpg")
    print("GPU Output -> gray_gpu.jpg")