#!/usr/bin/python3

import numpy as np
import torch

from torch import nn
from typing import Tuple



SCHARR_X_KERNEL = np.array(
    [
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ]).astype(np.float32)
SCHARR_Y_KERNEL = np.array(
    [
        [-3, -10, -3],
        [0, 0, 0],
        [3, 10, 3]
    ]).astype(np.float32)


def compute_image_gradients(image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use convolution with Scharr filters to compute the image gradient at each
    pixel.

    Args:
        image_bw: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    h, w=image_bw.shape #Gain the dimensions of image

    #zero-padding of the image
    padded_row=int((SCHARR_X_KERNEL.shape[0]-1)/2)
    padded_column=int((SCHARR_X_KERNEL.shape[1]-1)/2)
    padded_image_np=np.pad(image_bw,((padded_row,padded_row),(padded_column,padded_column)),'constant',constant_values=(0,0))
    
    
    #create container for gradient
    Ix=np.zeros((h,w),dtype=np.double)
    Iy=np.zeros((h,w),dtype=np.double)

 
    #Flip the kernal upside down and left side    
    flipped_filter_x = np.flipud(np.fliplr(SCHARR_X_KERNEL))
    flipped_filter_y = np.flipud(np.fliplr(SCHARR_Y_KERNEL))
    #print(f"Shape of filter {flipped_filter.shape}")

    #convolution process
    for i in range(h): #each row in image
        for j in range(w):  #each column in mage
            ROI=padded_image_np[padded_row+i-padded_row:padded_row+i+padded_row+1,padded_column+j-padded_column:padded_column+j+padded_column+1]
            Ix[i,j]=np.sum(ROI*flipped_filter_x)
            Iy[i,j]=np.sum(ROI*flipped_filter_y)
    
    Ix=np.float32(Ix)
    Iy=np.float32(Iy)

    return -Ix, -Iy
    raise NotImplementedError('`compute_image_gradients` function in ' +
        '`part1_harris_corner.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################




def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel.

    Args:
        ksize: dimension of square kernel
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel
    """

    norm_mu = int(ksize / 2)
    idxs_1d = torch.arange(ksize).float()
    exponents = -((idxs_1d - norm_mu) ** 2) / (2 * (sigma ** 2))
    gauss_1d = torch.exp(exponents)

    # make normalized column vector
    gauss_1d = gauss_1d.reshape(-1, 1) / gauss_1d.sum()
    gauss_2d = gauss_1d @ torch.transpose(gauss_1d, 0, 1)
    kernel = gauss_2d

    return kernel


def second_moments(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, then mixed derivatives,
    then compute the second moments (sx2, sxsy, sy2) at each pixel, using
    convolution with a Gaussian filter.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter

    Returns:
        sx2: array of shape (M,N) containing the second moment in the x direction
        sy2: array of shape (M,N) containing the second moment in the y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the y direction
    """

 
    Ix, Iy = compute_image_gradients(image_bw)

    Ix = torch.from_numpy(Ix)
    Iy = torch.from_numpy(Iy)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # combine along a new dimension
    channel_products = torch.stack((Ix2, Iy2, Ixy), 0).unsqueeze(0)

    # create second moments S_xx, S_yy and S_xy from I_xx, I_xy, I_yy
    Gk = get_gaussian_kernel_2D_pytorch(ksize=ksize, sigma=sigma)


    pad_size = ksize // 2
    conv2d_gauss = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=ksize,
        bias=False,
        padding=(pad_size, pad_size),
        padding_mode='zeros',
        groups=3
    )

    conv2d_gauss.weight = nn.Parameter(
        Gk.expand((3, 1, ksize, ksize))
    )
    second_moments = conv2d_gauss(channel_products)

    # compute corner responses
    sx2 = second_moments[:, 0, :, :].squeeze()
    sy2 = second_moments[:, 1, :, :].squeeze()
    sxsy = second_moments[:, 2, :, :].squeeze()

    sx2 = sx2.detach().numpy()
    sy2 = sy2.detach().numpy()
    sxsy = sxsy.detach().numpy()

    return sx2, sy2, sxsy


def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * in equation S_xx = Gk * I_xx is a convolutional operation over a
    Gaussian kernel of size (k, k).
    You may call the second_moments function above to get S_xx S_xy S_yy in M.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score

    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    #Gain the Harris Matrix
    sx2, sy2, sxsy = second_moments(image_bw, ksize, sigma)
    
    # Harris response map: R = det(M) - alpha * (trace(M))^2
    # det(M) = (S_xx * S_yy) - (S_xy^2), it is a element by element product, so that value for each pixel can be calculated
    det_M = (sx2 * sy2) - (sxsy ** 2)
    # trace(M) = S_xx + S_yy
    trace_M = sx2 + sy2
    # Harris response R
    R = det_M-alpha*(trace_M**2)

    
    return R
    raise NotImplementedError('`compute_harris_response_map` function in ' +
        '`part1_harris_corner.py` needs to be implemented')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################




def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d
            score/response map
    """

    dim_R = np.shape(R)
    maxpooled_R = np.zeros(dim_R)

    pad_x = ksize // 2
    pad_y = ksize // 2

    R_padded = np.pad(
        R,
        pad_width=((pad_x, pad_x), (pad_y, pad_y)),
        mode="constant",
        constant_values=(0, 0),
    )

    for i in range(0, dim_R[0]):
        for j in range(0, dim_R[1]):
            maxpooled_R[i, j] = R_padded[ i:i+ksize, j:j+ksize].max()

    return maxpooled_R


import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get top k interest points that are local maxima over (ksize, ksize)
    neighborhood.

    Args:
        R: score response map of shape (M, N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator

    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    padded_size=int((ksize-1)/2)
    # Convert numpy array to torch tensor
    R_torch = torch.from_numpy(R).float()
  
    #Make from 2D into 4D, so that MaxPool2D can process
    R_torch = torch.unsqueeze(R_torch, 0)  # Adds batch dimension
    R_torch = torch.unsqueeze(R_torch, 0)  # Adds channel dimension

   
    #Set thredhold
    Median=torch.median(R_torch)
    #Compare the value with Median one by one, and return the location, where value is larger than median
   
    #Set points to be 0, for those are smaller than median
    R_torch[R_torch<Median]=0

    #Go through maxpool2d to pick locally maximum point
    max_pool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=padded_size)
    R_in_pool = max_pool(R_torch)

    #Keep only the points which are local maxima
    R_torch= R_torch*(R_torch == R_in_pool)

    # Flatten the 2D response map to 1D
    R_flat = R_torch.view(-1)
    
    # Get the top k points based on their scores
    confidences, index = torch.topk(R_flat, k)

    # Convert flat indices to 2D indices (x, y coordinates)
    # X=floor(index/column_dims)
    # Y=index%column_dims
    y_index,x_index=torch.div(index, R_torch.shape[-1], rounding_mode='floor'),index % R_torch.shape[-1]

    # Convert tensors to numpy arrays
    x = x_index.cpu().numpy()
    y = y_index.cpu().numpy()
    confidences = confidences.cpu().numpy()

    return x,y,confidences
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    
    


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,) representing x coord of interest points
        y: array of shape (k,) representing y coord of interest points
        c: array of shape (k,) representing confidences of interest points

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    img_h, img_w = img.shape[0], img.shape[1]

    x_valid = (x >= 7) & (x <= img_w - 9)
    y_valid = (y >= 7) & (y <= img_h - 9)
    valid_idxs = x_valid & y_valid
    x,y,c = x[valid_idxs], y[valid_idxs], c[valid_idxs]

    return x, y, c


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500 #default to be 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the Harris Corner detector. You will find compute_harris_response_map(),
    nms_maxpool_pytorch(), and remove_border_vals() useful.
    Make sure to normalize your response map to fall within the range [0,1].
    The kernel size here is 7x7.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: maximum number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        c: array of dim (p,) containing the strength (confidence) of each
            interest point where p <= k.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    R=compute_harris_response_map(image_bw) #Gain R value for each pixel
    
    #Normalize the value of R first
    R_max=R.max()
    R_min=R.min()
    R=(R-R_min)/(R_max-R_min)

    x,y,c=nms_maxpool_pytorch(R,k,7) #Select k-top local maximum R
    x,y,c=remove_border_vals(image_bw,x,y,c) #Remove those points that are too close to boundary
    

    return x, y, c
    raise NotImplementedError('`get_harris_interest_points` function in ' +
        '`part1_harris_corner.py` needs to be implemented')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    
