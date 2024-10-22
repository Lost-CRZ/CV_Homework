#!/usr/bin/python3

import copy
import pdb
import time
from typing import Tuple
import numpy as np

import torch

from src.vision.part1_harris_corner import compute_image_gradients
from torch import nn


"""
Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells.
"""


def get_orientations_and_magnitudes(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the orientations and magnitudes of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    magnitudes=np.sqrt(Ix*Ix+Iy*Iy)
    
    sign=np.sign(Iy)
    # Replace zeros with 1
    sign[sign==0]=1

    size_row=np.shape(Ix)[0]
    size_coloumn=np.shape(Ix)[1]
    orientations=np.zeros([size_row,size_coloumn])
    orientations[magnitudes!=0]=sign[magnitudes!=0]*np.arccos(Ix[magnitudes!=0]/magnitudes[magnitudes!=0])
    

    return orientations, magnitudes
    raise NotImplementedError('`get_orientations_and_magnitudes` function ' +
        'in `part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    


def get_gradient_histogram_vec_from_patch(
    window_orientations: np.ndarray,
    window_magnitudes: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram.

    Args:
        window_orientations: (16,16) array representing gradient orientations of
            the patch
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    #bin edge from -pi to pi
    standard_bin_edge=np.linspace(-np.pi,np.pi,9)

    grid_width=window_magnitudes.shape[0]//4

    grid_orientation=np.zeros([grid_width,grid_width])
    grid_magnitude=np.zeros([grid_width,grid_width])

    wgh=np.array([])
    for i in range(4):
        for j in range(4):
            #Substract the 4*4 grid from window
            grid_orientation=window_orientations[i*grid_width:i*grid_width+grid_width,j*grid_width:j*grid_width+grid_width]
            grid_magnitude=window_magnitudes[i*grid_width:i*grid_width+grid_width,j*grid_width:j*grid_width+grid_width]

            #Convert 2D vector [grid_width,gridth]---->1D vector [grid_width^2], easier for following process
            grid_magnitude_flattened=grid_magnitude.flatten() 
            grid_orientation_flattened=grid_orientation.flatten()

            #create the histogram for each grid, here, it counts how many gradient vector fall in the 
            #sections according to their angle, with gradient magnitude as weights
            #the section is segmented by bins like [1,5], from 1 to 5 is section, 1 and 5 are bins
            diagram_value,diagram_bins=np.histogram(grid_orientation_flattened,bins=standard_bin_edge,weights=grid_magnitude_flattened)
            
   
            wgh=np.append(wgh,diagram_value)
        
    wgh=wgh.reshape(128,1)

    return wgh

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    


def get_feat_vec(
    x: float,
    y: float,
    orientations,
    magnitudes,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.

    Args:
        x: a float, the column (x-coordinate) of the interest point
        y: A float, the row (y-coordinate) of the interest point
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    delta = feature_width//2 - (feature_width+1)%2 #!!!!!! pay attion to the shape
    start_y = y - delta
    start_x = x -delta
    if start_y<0 or start_x<0 or start_y+feature_width>orientations.shape[0] or start_x+feature_width>orientations.shape[1]:
        print("warning: the index exceed the boundary in get_feat_vec")
    patch_orientations = orientations[start_y:start_y+feature_width, start_x:start_x+feature_width]
    patch_magnitudes = magnitudes[start_y:start_y+feature_width, start_x:start_x+feature_width]
    fv = get_gradient_histogram_vec_from_patch(patch_orientations, patch_magnitudes)
  
    fv = fv / (np.linalg.norm(fv, ord=2)+1e-12) 
    fv = np.sqrt(fv)

    

    return fv
    raise NotImplementedError('`get_feat_vec` function in ' +
        '`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    #Gain the gradient firts
    Ix,Iy=compute_image_gradients(image_bw)

    #Gain the orientation and magnitudes across the image
    orientations, magnitudes=get_orientations_and_magnitudes(Ix,Iy)

    #Gain the feat_dim for each point
    fvs=[]
    
    for x,y in zip(X,Y):

        rows=get_feat_vec(x,y,orientations,magnitudes,feature_width)
        rows=rows.flatten()
        fvs.append(rows)
        
    fvs = np.array(fvs)
    fvs = np.round(fvs, decimals=3)
    


    return fvs
    raise NotImplementedError('`get_SIFT_descriptors` function in ' +
        '`part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    


### ----------------- OPTIONAL (below) ------------------------------------

## Implementation of the function below is  optional (extra credit)


def get_sift_features_vectorized(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    This function is a vectorized version of `get_SIFT_descriptors`.

    As before, start by computing the image gradients, as done before. Then
    using PyTorch convolution with the appropriate weights, create an output
    with 10 channels, where the first 8 represent cosine values of angles
    between unit circle basis vectors and image gradient vectors at every
    pixel. The last two channels will represent the (dx, dy) coordinates of the
    image gradient at this pixel. The gradient at each pixel can be projected
    onto 8 basis vectors around the unit circle

    Next, the weighted histogram can be created by element-wise multiplication
    of a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
    tensor, where a tensor cell is activated if its value represents the
    maximum channel value within a "fibre" (see
    http://cs231n.github.io/convolutional-networks/ for an explanation of a
    "fibre"). There will be a fibre (consisting of all channels) at each of the
    (M,N) pixels of the "feature map".

    The four dimensions represent (N,C,H,W) for batch dim, channel dim, height
    dim, and weight dim, respectively. Our batch size will be 1.

    In order to create the 4d binary occupancy tensor, you may wish to index in
    at many values simultaneously in the 4d tensor, and read or write to each
    of them simultaneously. This can be done by passing a 1D PyTorch Tensor for
    every dimension, e.g., by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

    Finally, given 8d feature vectors at each pixel, the features should be
    accumulated over 4x4 subgrids using PyTorch convolution.

    You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
    flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
    torch.norm() helpful.

    Returns:
        fvs
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    H,W=np.shape(image_bw)

    image_tensor = torch.tensor(image_bw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)

    # Define Sobel filters for gradient calculation (dx, dy)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Compute image gradients
    Ix = torch.nn.functional.conv2d(image_tensor, sobel_x, padding=1)  # Gradient along x-axis
    Iy = torch.nn.functional.conv2d(image_tensor, sobel_y, padding=1)  # Gradient along y-axis

    # Compute gradient magnitude and orientation
    magnitudes = torch.sqrt(Ix**2 + Iy**2)
    orientations = torch.arctan2(Iy, Ix)

    angles = torch.linspace(-np.pi*7/8, np.pi*7/8, 8)

    #calculate the cos_project of the gradients of photos in different angle
    i=0
    cos_projections=torch.zeros(8,H,W)
    for i in range(8):
        #channel projections in i-th projections
        chaneel_projections=torch.cos(orientations-angles[i])
        cos_projections[i,:,:]=chaneel_projections


    feature_map = torch.zeros((1, 10, H, W), dtype=torch.float32)

    feature_map[:, :8, :, :] = cos_projections
    feature_map[:, 8, :, :] = Ix
    feature_map[:, 9, :, :] = Iy

    #gain the bool matrix, in other word, find which sections, should the tensor fall into
    #torch.max, find the maximum value of each pixels across channel, so that find where the gradient orientations should 
    #fall in region of standard_angles. Then only remains values of that pixels in the maximum channel
    #After that, use the return value of each channel to compare with original cos_projects, therefore
    #create 1/0 bool matrix, like a bin_diagram, therefore, when the magnitudes*bool_matrix, only maximum magnitude value along 
    # channel dimensions of each pixels will remains
    occupancy_tensor = (cos_projections == torch.max(cos_projections, dim=0, keepdim=True)[0])
    weighted_histogram = (magnitudes * occupancy_tensor)
  
    fvs = []
    feature_width = 16
    half_feature_width = feature_width//2 - (feature_width+1)%2
    grid_size = 4

    #8,1,H,W-> 8 kernal-channels, 1 input channel
    weight_kernal = torch.ones(8, 1, grid_size, grid_size, dtype=torch.float32)
    for x, y in zip(X, Y):
        start_y = y - half_feature_width
        start_x = x -half_feature_width

        #Substract the feature_window centers at each interested points
        Substracted_Window = weighted_histogram[:, :, start_y:start_y+feature_width, start_x:start_x+feature_width]
        #groups=8, divide the channels of kernal into 8 groups,then each channel of images will go convolutions of 
        #corresponding channel of kernal, like kernal[i,1,:,:]*image[1,i,:,:], each channel outputs [1,1,H,W] (one angle)
        #totally 8 channel, therefore, the ultimate result is [1,8,H,W]
        fv = nn.functional.conv2d(Substracted_Window, weight_kernal, stride=grid_size, groups=8)#[1,8,feature_width/4,feature_width/4]
        fv=fv.flatten()#8*4*4=128
        fv = fv / (torch.norm(fv)) 
        fv = torch.sqrt(fv)
        fvs.append(fv)

    fvs = torch.stack(fvs).squeeze().numpy()
   
    return fvs
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


