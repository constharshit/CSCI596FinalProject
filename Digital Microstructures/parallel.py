import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
from mpi4py import MPI

# MPI Initialization
comm = MPI.COMM_WORLD
current_rank = comm.Get_rank()
process_size = comm.Get_size()

sizeGrid = 256  
num_grains = 1000  # Number of grains

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

# Function to find the indices of the nearest non-zero pixel (i.e., nucleation site)
def find_nearest_nonzero_index(matrix, x, y):
    """
    Finds the indices of the nearest non-zero pixel in the matrix.

    Parameters:
    - matrix: The input matrix.
    - x, y: Coordinates of the pixel to find the nearest non-zero pixel.

    Returns:
    - row_index, col_index: Indices of the nearest non-zero pixel.
    """
    temp = matrix[x, y]
    matrix[x, y] = 0
    rows, cols = np.nonzero(matrix)
    matrix[x, y] = temp
    min_index = ((rows - x) ** 2 + (cols - y) ** 2).argmin()
    return rows[min_index], cols[min_index]

if current_rank == 0:
    # Initialize the digital microstructure as an nxn grid
    microstructure = np.zeros((sizeGrid, sizeGrid), dtype=np.int32)
    # Create num_grains nucleation sites in the microstructure by assigning the
    # first num_grains whole numbers to random pixels
    microstructure[np.random.randint(sizeGrid, size=num_grains),
                   np.random.randint(sizeGrid, size=num_grains)] = np.arange(1, num_grains + 1)
    # Visualize the microstructure after the generation of num_grains nucleation sites
    visualize_heatmap(microstructure, 'parallel1.png')
    
    # Replace 0 pixels with the value of the nearest non-zero pixel
    updated_microstructure = np.copy(microstructure)
    for i in range(sizeGrid):
        for j in range(sizeGrid):
            if microstructure[i, j] == 0:
                row_index, col_index = find_nearest_nonzero_index(microstructure, i, j)
                updated_microstructure[i, j] = microstructure[row_index, col_index]
    visualize_heatmap(updated_microstructure, 'parallel2.png')

# Function to calculate the grain boundary pixels
def calculate_grain_boundary_pixels(matrix, rank):
    """
    Calculates the number of grain boundary pixels in a subgrid assigned to a processor.

    Parameters:
    - matrix: The subgrid assigned to the processor.
    - rank: The rank of the processor.

    Returns:
    - grain_boundary_pixels: The number of grain boundary pixels in the subgrid.
    """
    grain_boundary_pixels = 0  # Grain boundary pixels
    ranges = np.array([[0, matrix.shape[0] - 1, 0, matrix.shape[0] - 1],
                       [1, matrix.shape[0], 0, matrix.shape[0] - 1],
                       [1, matrix.shape[0], 1, matrix.shape[0]],
                       [1, matrix.shape[0], 0, matrix.shape[0] - 1]])
    for i in range(ranges[rank, 0], ranges[rank, 1]):
        for j in range(ranges[rank, 2], ranges[rank, 3]):
            if i == 0:
                if j == 0:
                    neighbors = matrix[0:2, 0:2] - matrix[i, j]
                elif j == matrix.shape[1] - 1:
                    neighbors = matrix[0:2, -2:] - matrix[i, j]
                else:
                    neighbors = matrix[0:2, j - 1:j + 2] - matrix[i, j]
            elif i == matrix.shape[0] - 1:
                if j == 0:
                    neighbors = matrix[-2:, 0:2] - matrix[i, j]
                elif j == matrix.shape[1] - 1:
                    neighbors = matrix[-2:, -2:] - matrix[i, j]
                else:
                    neighbors = matrix[-2:, j - 1:j + 2] - matrix[i, j]
            elif j == 0:
                neighbors = matrix[i - 1:i + 2, 0:2] - matrix[i, j]
            elif j == matrix.shape[1] - 1:
                neighbors = matrix[i - 1:i + 2, -2:] - matrix[i, j]
            else:
                neighbors = matrix[i - 1:i + 2, j - 1:j + 2] - matrix[i, j]
            # For grain interior pixels, the sum of neighbors is zero
            # For grain boundary pixels, the sum of neighbors is non-zero
            if np.sum(neighbors) != 0:
                grain_boundary_pixels += 1
    return grain_boundary_pixels

# Divide the grid into 4 sub-grids and send each sub-grid to a processor along with its neighboring row and column
if current_rank == 0:
    n1 = int(sizeGrid / 2)
    subgrid = updated_microstructure[0:n1 + 1, 0:n1 + 1]
    comm.send(updated_microstructure[n1 - 1:, 0:n1 + 1], dest=1, tag=1)
    comm.send(updated_microstructure[n1 - 1:, n1 - 1:], dest=2, tag=1)
    comm.send(updated_microstructure[0:n1 + 1, n1 - 1:], dest=3, tag=1)
else:
    subgrid = comm.recv(source=0, tag=1)

comm.Barrier()
wall_time = MPI.Wtime()

# For every sub-grid, compute the number of grain boundary points
grain_boundary_points = calculate_grain_boundary_pixels(subgrid, current_rank)
total_grain_boundary_points = comm.reduce(grain_boundary_points, op=MPI.SUM, root=0)

wall_time = MPI.Wtime() - wall_time

if current_rank == 0:
    print("Execution time = {t} ms".format(t=wall_time))  
    fraction_gb = total_grain_boundary_points / (np.square(sizeGrid))
    print("Fraction of grain boundary pixels = {frac}".format(frac=fraction_gb))
