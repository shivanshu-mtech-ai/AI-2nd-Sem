import math
import os
import numpy as np
from numba import cuda
from PIL import Image

@cuda.jit
def rgb_to_gray(rgb_img, gray_img):
    x, y = cuda.grid(2)

    if x < rgb_img.shape[0] and y < rgb_img.shape[1]:
        r = rgb_img[x, y, 0]
        g = rgb_img[x, y, 1]
        b = rgb_img[x, y, 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray_img[x, y] = gray


def process_single_image(input_path, output_path):

    img = Image.open(input_path).convert("RGB")
    rgb = np.array(img).astype(np.float32)

    height, width = rgb.shape[:2]
    gray = np.zeros((height, width), dtype=np.float32)

    d_rgb = cuda.to_device(rgb)
    d_gray = cuda.to_device(gray)

    threads_per_block = (16, 16)
    blocks_x = math.ceil(height / threads_per_block[0])
    blocks_y = math.ceil(width / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)

    rgb_to_gray[blocks_per_grid, threads_per_block](d_rgb, d_gray)

    gray_result = d_gray.copy_to_host()

    gray_uint8 = np.clip(gray_result, 0, 255).astype(np.uint8)
    Image.fromarray(gray_uint8).save(output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    # Get directory where this script exists
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Input image paths
    image1 = os.path.join(BASE_DIR, "images", "input_1.jpg")
    image3 = os.path.join(BASE_DIR, "images", "input_3.jpg")
    image4 = os.path.join(BASE_DIR, "images", "input_4.jpg")


    output1 = os.path.join(BASE_DIR, "images", "gray_output1.jpg")
    output3 = os.path.join(BASE_DIR, "images", "gray_output3.jpg")
    output4 = os.path.join(BASE_DIR, "images", "gray_output4.jpg")

    process_single_image(image1, output1)
    process_single_image(image3, output3)
    process_single_image(image4, output4)

    print("All images processed successfully!")