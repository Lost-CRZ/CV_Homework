#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def my_conv2d_pytorch(image: torch.Tensor, kernel: torch.Tensor, stride: int = 1, dilation: int = 1) -> torch.Tensor:
    """
    Applies input filter(s) to the input image.

    Args:
        image: Tensor of shape (1, d1, h1, w1)
        kernel: Tensor of shape (N, d1/groups, k, k) to be applied to the image
        stride: Stride of the filter
        dilation: Dilation of the filter
    Returns:
        filtered_image: Tensor of shape (1, d2, h2, w2) where
           d2 = N
           k' = k * d1 - d1 + 1
           h2 = (h1 - k' + 2 * padding) / stride + 1
           w2 = (w1 - k' + 2 * padding) / stride + 1

    HINTS:
    - You should use the 2d convolution operator from torch.nn.functional.
    - In PyTorch, d1 is `in_channels`, and d2 is `out_channels`
    - Make sure to pad the image appropriately (it's a parameter to the
      convolution function you should use here!).
    - You can assume the number of groups is equal to the number of input channels.
    - You can assume only square filters for this function.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    # Get dimensions of input image and kernel
    d1 = image.size(1)  
    h1 = image.size(2) 
    w1 = image.size(3)

    N=kernel.size(0)
    group_num = d1//kernel.size(1)
    k = kernel.size(2) 

    # Calculate padding for "same" output size
    k_new=k*dilation-dilation+1
    padding = ((k_new - dilation) // 2) 
    d2 = kernel.size(0) 
    h2=(h1-k_new+2*padding)//stride+1
    w2=(w1-k_new+2*padding)//stride+1

    image=torch.nn.functional.pad(image,[padding,padding,padding,padding],mode='constant',value=0.0)
    # Apply 2D convolution using PyTorch's functional API
    filtered_image = torch.nn.functional.conv2d(
        image,
        kernel,
        stride=stride,
        dilation=dilation,
        groups=group_num
    )

    return filtered_image
    #raise NotImplementedError(
    #    "`my_conv2d_pytorch` function in `part3.py` needs to be implemented"
    #)

    ### END OF STUDENT CODE ####
    ############################

   
