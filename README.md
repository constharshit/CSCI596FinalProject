# Project-Final
PARALLEL Contribution by Ritvik, Rohit and Harshit.

# Facial Recognition Project

## Project Overview

This facial recognition project utilizes the DigiFace1M dataset, a collection of over one million diverse synthetic face images designed for face recognition applications. The goal of this project is to train a facial recognition model that can accurately identify and classify faces in real-world scenarios.



[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)

# Digital Microstructure Evolution with MPI Parallelization
In serial.py, the simulation is carried out sequentially, with additional features to visualize the microstructure, replace pixels, and measure grain boundary fractions. On the other hand, parallel.py leverages MPI (Message Passing Interface) for distributed computing, enabling parallelized simulations and efficient calculation of grain boundary pixels. Explore each script to understand the evolution process, visualize results, and compare the execution times between sequential and parallel approaches.



# Working of parallel.py

`parallel.py` simulates the `evolution of a digital microstructure and calculates the fraction of grain boundary pixels`. The simulation is parallelized using `MPI (Message Passing Interface)` for distributed computing.

The program simulates the evolution of a digital microstructure in a parallelized manner using MPI, calculates grain boundary pixels, and provides timing and output information. The parallelization allows for distributed processing of the microstructure, improving efficiency for large-scale simulations.

Initialization :
   - The program starts by initializing MPI communication, getting the current rank, and the total number of processes.
   - It sets the size of the grid (`sizeGrid`) and the number of nucleation sites (`num_grains`).
   - A digital microstructure grid (`microstructure`) is created with nucleation sites randomly assigned integer values.
   - The initial microstructure is visualized as a heatmap and saved to a file (`parallelMicro.png`).

Nucleation Site Replacement :
   - The program creates a copy of the microstructure (`updated_microstructure`) and iterates over each pixel with a value of 0.
   - For each 0 pixel, it replaces the value with the value of the nearest non-zero pixel in the microstructure.
   - The updated microstructure is visualized as a heatmap and saved to a file (`parallelMicro2.png`).

Grain Boundary Calculation Function :
   - There is a function (`calculate_grain_boundary_pixels`) that calculates the number of grain boundary pixels in a given subgrid assigned to a processor.
   - It considers neighboring rows and columns to determine grain boundary pixels.

Grid Division and Communication :
   - The grid is divided into four subgrids, and each subgrid is sent to a different processor (Rank 1, 2, 3).
   - Each processor also receives the neighboring rows and columns of its subgrid from the master process.

Grain Boundary Calculation :
   - Each processor calculates the number of grain boundary pixels in its assigned subgrid using the `calculate_grain_boundary_pixels` function.
   - The results are then reduced using MPI's `comm.reduce` operation to obtain the total number of grain boundary pixels.

Timing and Output :
   - The program measures the execution time using MPI's wall time.
   - The execution time and the fraction of grain boundary pixels are printed by the master process.



# Working of serial.py

`serial.py` simulates the evolution of a digital microstructure, visualize it, replace certain pixels with their nearest non-zero neighbors, calculate the fraction of grain boundary pixels, and measure the execution time. 

Initialization :
   -  A grid (`arr`) of size `gridSize x gridSize` is initialized with zeros.
   - `numGrains` nucleation sites are randomly assigned whole numbers in the grid.

Visualization (Heatmap) :
   - The initial microstructure is visualized as a heatmap and saved to a file (`serial1.png`).

Nearest Non-Zero Pixel Replacement : 
   - A copy of the grid (`arr1`) is created.
   - For each pixel with a value of 0, it is replaced with the value of its nearest non-zero neighbor.
   - The updated microstructure is visualized as a heatmap and saved to a file (`serial2.png`).

 Grain Boundary Calculation :
   - A function (`calculate_fraction_of_grain_boundary_pixels`) is defined to calculate the fraction of grain boundary pixels in the matrix.
   - It considers the neighbors of each pixel and checks if the sum of neighbors is non-zero.
   - The fraction of grain boundary pixels is calculated as the ratio of grain boundary pixels to the total number of pixels.

Execution Time Measurement :
   - The script measures the execution time of the grain boundary calculation.
   - The start and end times are recorded, and the difference is multiplied by 10^3 to convert seconds to milliseconds.

Output :
   - The execution time and the fraction of grain boundary pixels are printed.


# Comparision between Serial and Parallel ( MPI ) plots

- With `Number of Grains` : 1000 (Parallel)
  ![Parallel Image](DigitalMicrostructures/n-1000/parallel1-1000.png) 
  ![Parallel Image](DigitalMicrostructures/n-1000/parallel2-1000.png)
- With `Number of Grains` : 1000 (Serial)
  ![Parallel Image](DigitalMicrostructures/n-1000/serial1-1000.png)
  ![Parallel Image](DigitalMicrostructures/n-1000/serial2-1000.png)
- Plot Comparision for `Number of Grains` : 1000
  ![Parallel Image](DigitalMicrostructures/n-1000/n1000.png)

- With `Number of Grains` : 500 (Parallel)
  ![Parallel Image](DigitalMicrostructures/n-500/parallel1-500.png)
  ![Parallel Image](DigitalMicrostructures/n-500/parallel2-500.png)
- With `Number of Grains` : 500 (Serial)
  ![Parallel Image](DigitalMicrostructures/n-500/serial1-500.png)
  ![Parallel Image](DigitalMicrostructures/n-500/serial2-500.png)
- Plot Comparision for `Number of Grains` : 500
  ![Parallel Image](DigitalMicrostructures/n-500/n500.png)

- With `Number of Grains` : 10000 (Parallel)
  ![Parallel Image](DigitalMicrostructures/n-10000/parallel1-10000.png)
  ![Parallel Image](DigitalMicrostructures/n-10000/parallel2-10000.png)
- With `Number of Grains` : 10000 (Serial)
  ![Parallel Image](DigitalMicrostructures/n-10000/serial1-10000.png)
  ![Parallel Image](DigitalMicrostructures/n-10000/serial2-10000.png)
- Plot Comparision for `Number of Grains` : 10000
  ![Parallel Image](DigitalMicrostructures/n-10000/n10000.png)








# Thank You
[(Back to top)](#Project-Final)



Professor Aichiiro Nakano
TA Taufeq
TA Shin


