import numpy as np
import random
import matplotlib.pyplot as plt
from time import time

gridSize = 256  # Grid size
numGrains = 10000  # Number of grains

# Initialize the digital microstructure as an nxn grid
arr = np.zeros((gridSize, gridSize), dtype=np.int32)

# Create numGrains nucleation sites in the microstructure by assigning the
# first numGrains whole numbers to random pixels
arr[np.random.randint(gridSize, size=numGrains), np.random.randint(gridSize, size=numGrains)] = np.arange(1, numGrains + 1)

# Function to visualize the grid as a heatmap
def visualize_heatmap(matrix, filename):
    """
    Visualizes the matrix as a heatmap and saves the plot to a file.

    Parameters:
    - matrix: The input matrix to visualize.
    - filename: The name of the file to save the heatmap plot.
    """
    plt.imshow(matrix)
    plt.xlim(0, matrix.shape[0] - 1)
    plt.ylim(0, matrix.shape[1] - 1)
    plt.colorbar()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Visualize the microstructure after the generation of numGrains nucleation sites
visualize_heatmap(arr, 'serial1.png')

# Function to find the indices of the nearest non-zero pixel (i.e., nucleation site)
def find_nearest_nonzero_index(a, x, y):
    """
    Finds the indices of the nearest non-zero pixel in the matrix.

    Parameters:
    - a: The input matrix.
    - x, y: Coordinates of the pixel to find the nearest non-zero pixel.

    Returns:
    - row_index, col_index: Indices of the nearest non-zero pixel.
    """
    tmp = a[x, y]
    a[x, y] = 0
    rows, cols = np.nonzero(a)
    a[x, y] = tmp
    min_index = ((rows - x) ** 2 + (cols - y) ** 2).argmin()
    return rows[min_index], cols[min_index]

# Replacing 0 pixels with the value of the nearest non-zero pixel
arr1 = np.copy(arr)
for i in range(gridSize):
    for j in range(gridSize):
        if arr[i, j] == 0:
            row_index, col_index = find_nearest_nonzero_index(arr, i, j)
            arr1[i, j] = arr[row_index, col_index]
visualize_heatmap(arr1, 'serial2.png')

# Function to calculate the grain boundary pixels
def calculate_fraction_of_grain_boundary_pixels(a):
    """
    Calculates the fraction of grain boundary pixels in the given matrix.

    Parameters:
    - a: The input matrix.

    Returns:
    - fraction: The fraction of grain boundary pixels.
    """
    grain_boundary_pixels = 0  # Grain boundary pixels
    rows, cols = a.shape[0], a.shape[1]
    for i in range(rows):
        for j in range(cols):
            if i == 0:
                if j == 0:
                    neighbors = a[0:2, 0:2] - a[i, j]
                elif j == cols - 1:
                    neighbors = a[0:2, -2:] - a[i, j]
                else:
                    neighbors = a[0:2, j - 1:j + 2] - a[i, j]
            elif i == rows - 1:
                if j == 0:
                    neighbors = a[-2:, 0:2] - a[i, j]
                elif j == cols - 1:
                    neighbors = a[-2:, -2:] - a[i, j]
                else:
                    neighbors = a[-2:, j - 1:j + 2] - a[i, j]
            elif j == 0:
                neighbors = a[i - 1:i + 2, 0:2] - a[i, j]
            elif j == cols - 1:
                neighbors = a[i - 1:i + 2, -2:] - a[i, j]
            else:
                neighbors = a[i - 1:i + 2, j - 1:j + 2] - a[i, j]
            # For grain interior pixels, the sum of neighbors is zero
            # For grain boundary pixels, the sum of neighbors is non-zero
            if np.sum(neighbors) != 0:
                grain_boundary_pixels += 1
    return grain_boundary_pixels / (rows * cols)

start_time = time()
fraction_gb = calculate_fraction_of_grain_boundary_pixels(arr1)
end_time = time()

execution_time = (end_time - start_time) * 10 ** 3  # execution time in milliseconds
print("Execution time = {t} ms".format(t=execution_time))
print("Fraction of grain boundary pixels = {frac}".format(frac=fraction_gb))
