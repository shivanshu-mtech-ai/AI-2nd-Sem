import os
import numpy as np
from PIL import Image
from numba import njit

# ================= DATA LOADER =================

IMG_SIZE = 32

def load_dataset(base_path):

    X, y = [], []

    classes = {
        "with_mask": 1,
        "without_mask": 0
    }

    for cname, label in classes.items():
        folder = os.path.join(base_path, cname)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            img = Image.open(path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))

            arr = np.array(img, dtype=np.float32) / 255.0

            X.append(arr)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ================= CNN LAYERS =================

@njit
def conv2d(image, kernel):
    h, w, c = image.shape
    kh, kw, _ = kernel.shape

    out = np.zeros((h-kh+1, w-kw+1), dtype=np.float32)

    for i in range(h-kh+1):
        for j in range(w-kw+1):
            s = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    for ch in range(c):
                        s += image[i+ki, j+kj, ch] * kernel[ki, kj, ch]
            out[i, j] = s

    return out


@njit
def relu(x):
    return np.maximum(x, 0)


@njit
def maxpool(x):
    h, w = x.shape
    out = np.zeros((h//2, w//2), dtype=np.float32)

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            out[i//2, j//2] = np.max(x[i:i+2, j:j+2])

    return out


@njit
def dense(x, W, b):
    return np.dot(x, W) + b


@njit
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SimpleCNN:

    def __init__(self):

        self.kernel = np.random.randn(3,3,3).astype(np.float32) * 0.1

        flattened = 15 * 15
        self.W = np.random.randn(flattened,1).astype(np.float32) * 0.1
        self.b = np.zeros((1,), dtype=np.float32)

    def forward(self, image):

        x = conv2d(image, self.kernel)
        x = relu(x)
        x = maxpool(x)

        x = x.flatten().astype(np.float32)

        x = dense(x, self.W, self.b)
        out = sigmoid(x)

        return out[0]

    def predict(self, image):

        prob = self.forward(image)

        if prob >= 0.5:
            return 1
        else:
            return 0


if __name__ == "__main__":

    DATASET_PATH = "" #input the dataset path here

    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)

    print("Dataset size:", len(X))

    model = SimpleCNN()

    correct = 0

    print("Running classification...")

    for i in range(len(X)):
        pred = model.predict(X[i])

        if pred == y[i]:
            correct += 1

    accuracy = correct / len(X)

    print("\n===== RESULT =====")
    print("Accuracy:", accuracy * 100, "%")