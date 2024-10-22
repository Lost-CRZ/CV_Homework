#!/usr/bin/python3
from typing import Tuple
from PIL import Image
import numpy as np
import vision.utils as utils


def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.

    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1

    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution

    Returns:
        kernel: 1d column vector of shape (k,1)

    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    kernel=np.zeros(ksize,dtype=float) #create 1D, ksize elements original array
    kernel=kernel.reshape(-1,1)
    mean=int(np.floor(ksize/2))
    
    i=0
    while i<ksize:
        exp_value=-(np.power(i-mean,2))/(2*sigma*sigma)
        kernel[i]=(1/(np.sqrt(2*np.pi)*sigma))*np.exp(exp_value) #the area between two points        
        i=i+1

    kernel=kernel/np.sum(kernel) #normalize dataset, ensuring the sum=1, data[i]/sum
    return kernel
    #raise NotImplementedError(
    #    "`create_Gaussian_kernel_1D` function in `part1.py` needs to be implemented")

    ### END OF STUDENT CODE ####
    ############################
    


def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    k=int(cutoff_frequency*4+1)
    kernel=np.zeros((k,k))
    mean=np.floor(k/2)
    sigma=cutoff_frequency

    kernel_x=np.zeros((k,1))
    kernel_x=create_Gaussian_kernel_1D(k,sigma)
    kernel_x=kernel_x.reshape(1,k)

    kernel_y=np.zeros((k,1))
    kernel_y=create_Gaussian_kernel_1D(k,sigma)
    
    kernel=np.outer(kernel_y,kernel_x)
    return kernel


    #raise NotImplementedError(
    #    "`create_Gaussian_kernel_2D` function in `part1.py` needs to be implemented"
    #)

    ### END OF STUDENT CODE ####
    ############################

  


def my_conv2d_numpy(image_path: str, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.

    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image_path: string specifying the path to the input image
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c) #m*n size and num of c channel

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices and functions from utils.py is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """
    #Open the image and store it into img_np
    img_np = utils.load_image(image_path)
    #img_np = np.array(img,dtype=np.float64)
    h, w, c = img_np.shape #Gain the dimensions of image

    filter_k=filter.shape[0] #row number of filter
    filter_j=filter.shape[1] #column number of filter
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

   
    #zero-padding of the image
    padded_row=int((filter_k-1)/2)
    padded_column=int((filter_j-1)/2)
    padded_image_np=np.pad(img_np,((padded_row,padded_row),(padded_column,padded_column),(0,0)),'constant',constant_values=(0,0))
    #print(f"Shape of Padded image {padded_image_np.shape}")
    #create container for filtered image
    filtered_img=np.zeros((h,w,c),dtype=np.float32)

    #print(f"Shape of filtered image {filtered_img.shape}")
    
    flipped_filter = np.flipud(np.fliplr(filter))
    #print(f"Shape of filter {flipped_filter.shape}")

    #convolution process
    for k in range(c): #each channel
      for i in range(h): #each row in image
        for j in range(w):  #each column in mage
            ROI=padded_image_np[padded_row+i-padded_row:padded_row+i+padded_row+1,padded_column+j-padded_column:padded_column+j+padded_column+1,k]
            filtered_img[i,j,k]=np.sum(ROI*flipped_filter)
          
    
    ############################
    ### TODO: YOUR CODE HERE ###
    return filtered_img
    #raise NotImplementedError(
    #    "`my_conv2d_numpy` function in `part1.py` needs to be implemented"
    #)

    ### END OF STUDENT CODE ####
    ############################



def create_hybrid_image(
    image_path1: str, image_path2: str, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image_path1: string specifying the path to the input image
        image_path2: string specifying the path to the input image
        filter: array of dim (x, y)
    Returns:
        low_frequencies: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    filter_kernal=create_Gaussian_kernel_2D(7)
    Identical_kernal=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    low_frequencies=my_conv2d_numpy(image_path1,filter_kernal)
    high_frequencies=my_conv2d_numpy(image_path2,Identical_kernal)-my_conv2d_numpy(image_path2,filter_kernal)
    hybrid_image=low_frequencies+high_frequencies
    hybrid_image=np.clip(hybrid_image,0,1)

    return low_frequencies, high_frequencies, hybrid_image
    #raise NotImplementedError(
    #    "`create_hybrid_image` function in `part1.py` needs to be implemented"
    #)

    ### END OF STUDENT CODE ####
    ############################

