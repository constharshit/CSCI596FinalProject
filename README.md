# Facial Recognition and Digital Microstructure Evolution

## Table of Contents
- [Facial Recognition using CUDA](#facial-recognition-using-cuda)
  - [Dataset](#dataset)
  - [Working of Facial Recognition using CUDA](#working-of-facial-recognition-using-cuda)
  - [Plots and Analysis with and without using CUDA](#plots-and-analysis-with-and-without-using-cuda)
- [Digital Microstructure Evolution with MPI Parallelization](#digital-microstructure-evolution-with-mpi-parallelization)
  - [Working of parallel.py](#working-of-parallelpy)
  - [Working of serial.py](#working-of-serialpy)
  - [Comparison between Serial and Parallel (MPI) Plots](#comparison-between-serial-and-parallel-mpi-plots)
- [Contributors](#contributors)
- [Thank You](#thank-you)

---

# Facial Recognition using CUDA run on multiple GPUs

This section of the repository focuses on facial recognition using CUDA technology and is a collaborative effort by Ritvik, Rohit, and Harshit.

## Dataset

The DigiFace-1M dataset is a collection of over one million diverse synthetic face images for face recognition. 

https://github.com/microsoft/DigiFace1M

DigiFace1M provides a large and varied set of synthetic face images. This dataset is essential for creating a robust model capable of handling diverse facial features, expressions, and lighting conditions.

The dataset contains:

720K images with 10K identities (72 images per identity). For each identity, 4 different sets of accessories are sampled and 18 images are rendered for each set.
500K images with 100K identities (5 images per identity). For each identity, only one set of accessories is sampled.
We are using one part of this dataset with 166K images.

This FaceDataset class is designed for creating a triplet dataset for training a facial recognition model. It inherits from the PyTorch Dataset class and takes a list of identity folders as input, where each folder contains images of a specific individual. During training, it randomly selects an anchor image, a positive image (belonging to the same identity as the anchor but different from it), and a negative image (belonging to a different identity) to form a triplet. The dataset is then used to train a neural network to learn facial embeddings in an unsupervised manner. This approach encourages the model to map faces of the same identity close together in the embedding space while keeping faces of different identities apart, contributing to improved facial recognition performance.

visualization of a triplet of images (Anchor, Positive, Negative) in a single row.

<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/dataset_images.png">
<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/dataset_image_1.png">

## Working of Facial Recognition using CUDA

This facial recognition project utilizes the DigiFace1M dataset, a collection of over one million diverse synthetic face images designed for face recognition applications. The goal of this project is to train a facial recognition model that can accurately identify and classify faces in real-world scenarios. 

We used a Siamese neural network architecture designed for facial recognition tasks. The model employs a MobileNetV3 backbone with frozen weights for efficient feature extraction. The classifier head is replaced with a custom configuration, consisting of two fully connected layers, aiming to reduce the feature dimensionality to a 256-dimensional vector. The model utilizes the L1 distance metric for computing distances between anchor, positive, and negative face embeddings during triplet loss computation. The final logistic layer produces a binary output indicating whether the input pair of faces belongs to the same identity (1) or different identities (0). This Siamese model is intended for training with triplet loss to enhance facial feature representation for improved facial recognition accuracy. We trained the model using Stochastic Gradient Descent (SGD) with a binary cross-entropy loss criterion. 

For training, we used Google Colab notebook and their T4 GPUs. We wrote code using PyTorch framework. We used CUDA to run the training on both the models.

#### Model Architecture
<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/model_arch.jpeg">


We used Principal Component Analysis (PCA) to reduce the dimensionality of image features from several triplets, converting them into 2D coordinates. It then plots a scatter plot where each point represents a triplet, and the color distinguishes different triplets. The plot helps visualize the distribution of triplets in a reduced feature space, providing insights into the relationships or separations between different triplets. The significance lies in understanding the structure of the data in a lower-dimensional representation, which can aid in identifying patterns or similarities among triplets.

<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/scatter_plot.png">

We generated bar chart plots for each image in three triplets. Each triplet consists of three bars representing the features of the anchor, positive, and negative images. The x-axis corresponds to different features, and the y-axis represents the values of these features. The significance lies in visually comparing the feature values across different images in each triplet, providing insights into how the features vary within and between triplets.

<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/bargraphs_layout.png">


## Plots and Analysis with and without using CUDA

Below is the training phase of our model implementing the parallelization. The code trains a Siamese neural network model using PyTorch. It utilizes parallelization to speed up training by distributing the workload across available GPUs. The model is optimized using Stochastic Gradient Descent, and its performance is evaluated over 10 epochs with periodic saving of checkpoints. The entire training process is timed, showing the duration with parallelization.

<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/epochs_parallel_2.png">
<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/epochs_parallel_1.png">

Initially, we loaded the data using a single worker and we trained the model for 10 epochs using a single GPU. We achieved a validation loss: 0.6589 and validation accuracy: 83.73%. It took a total of 2290.8079085350037s to train the model.

<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/model1_res.png">

Later, we performed parallelization for the above code. We used four workers to load the data and we trained the model for 10 epochs on 2 T4 GPUs parallely. We utilized PyTorch and DataLoader to handle the training data, with support for multi-GPU training using nn.DataParallel on 2 T4 GPUs from Google Colab. We achieved a validation loss: 0.6116 and validation accuracy: 87.01%. It took a total of 1845.6194655895233s to train the model.

<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/model2_res.png">

In conclusion, we got a better performing model with 19.43% improvement in speed.


The bar chart visually compares the execution time of a model with and without parallelization. The blue bar represents the time taken without parallelization, while the orange bar represents the time with parallelization. The chart illustrates the efficiency improvement achieved by parallelizing the model, with the orange bar showing a shorter execution time.

<img src="https://github.com/constharshit/CSCI596FinalProject/blob/master/FaceRecognition/time_comparison.png">


---

# Digital Microstructure Evolution with MPI Parallelization

In this section, the team explores the digital microstructure evolution using MPI (Message Passing Interface) parallelization. Three contributors, Ritvik, Rohit and Harshit have developed `parallel.py` and `serial.py` to simulate and visualize the evolution process.

## Working of parallel.py

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

## Working of serial.py

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

## Comparison between Serial and Parallel (MPI) Plots

The comparison plots showcase the performance differences between the serial and parallel approaches for different numbers of grains in the microstructure.

<div style="display:flex; justify-content: space-between;">
  <div>
    <p><strong> With `Number of Grains` : 500 (Parallel)</strong></p>
    <img src="DigitalMicrostructures/n-500/parallel1-500.png" alt="Parallel Image 1" width="400"/>
    <img src="DigitalMicrostructures/n-500/parallel2-500.png" alt="Parallel Image 2" width="400"/>
  </div>
  
  <div>
    <p><strong>With `Number of Grains` : 500 (Serial)</strong></p>
    <img src="DigitalMicrostructures/n-500/serial1-500.png" alt="Serial Image 1" width="400"/>
    <img src="DigitalMicrostructures/n-500/serial2-500.png" alt="Serial Image 2" width="400"/>
  </div>

  <div>
    <p><strong>Comaprision for Number of Grains: 500</strong></p>
    <img src="DigitalMicrostructures/n-500/n500.png" alt="Comparision500" />
  </div>
</div>
<p></p>
<div style="display:flex; justify-content: space-between;">
  <div>
    <p><strong>With `Number of Grains` : 1000 (Parallel)</strong></p>
    <img src="DigitalMicrostructures/n-1000/parallel1-1000.png" alt="Parallel Image 1" width="400"/>
    <img src="DigitalMicrostructures/n-1000/parallel2-1000.png" alt="Parallel Image 2" width="400"/>
  </div>
  
  <div>
    <p><strong>With `Number of Grains` : 1000 (Serial)</strong></p>
    <img src="DigitalMicrostructures/n-1000/serial1-1000.png" alt="Serial Image 1" width="400"/>
    <img src="DigitalMicrostructures/n-1000/serial2-1000.png" alt="Serial Image 2" width="400"/>
  </div>

  <div>
    <p><strong>Comaprision for Number of Grains: 1000</strong></p>
    <img src="DigitalMicrostructures/n-1000/n1000.png" alt="Comparision1000" />
  </div>
</div>
<p></p>
<div style="display:flex; justify-content: space-between;">
  <div>
    <p><strong>With `Number of Grains` : 10000 (Parallel)</strong></p>
    <img src="DigitalMicrostructures/n-10000/parallel1-10000.png" alt="Parallel Image 1" width="400"/>
    <img src="DigitalMicrostructures/n-10000/parallel2-10000.png" alt="Parallel Image 2" width="400"/>
  </div>
  
  <div>
    <p><strong>With `Number of Grains` : 10000 (Serial)</strong></p>
    <img src="DigitalMicrostructures/n-10000/serial1-10000.png" alt="Serial Image 1" width="400"/>
    <img src="DigitalMicrostructures/n-10000/serial2-10000.png" alt="Serial Image 2" width="400"/>
  </div>
  
  <div>
    <p><strong>Comaprision for Number of Grains: 10000</strong></p>
    <img src="DigitalMicrostructures/n-10000/n10000.png" alt="Comparision10000" />
  </div>
</div>


# Contributors

- Ritvik Nimmagadda
- Venkata Rohit Kumar Kandula
- Harshit Kumar Jain

# Thank You
[(Back to top)](#facial-recognition-and-digital-microstructure-evolution)

「ありがとうございます、先生」
