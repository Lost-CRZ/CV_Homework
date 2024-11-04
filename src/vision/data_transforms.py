"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fundamental_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    # Define the sequence of transforms
    fundamental_transforms = transforms.Compose([
        transforms.Resize(inp_size),          # Resize to the specified input size
        transforms.ToTensor(),                # Convert the image to a PyTorch tensor
        #transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    ])
    
    
    return fundamental_transforms
    raise NotImplementedError(
        "`get_fundamental_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ##############################9#############################################
    


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    fund_aug_transforms = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    fund_aug_transforms = transforms.Compose([
        transforms.Resize(inp_size),          # Resize to the specified input size
        
        # Apply color jitter to adjust brightness, contrast, saturation, and hue
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Randomly flip the image horizontally
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),                # Convert the image to a PyTorch tensor
        #transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    ])
    
    return fund_aug_transforms
    raise NotImplementedError(
        "`get_fundamental_augmentation_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code end
    ###########################################################################
    


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model alomg with
    normalization.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fund_norm_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################
    
    # Define the transformations
    fund_norm_transforms = transforms.Compose([
        transforms.Resize(inp_size),               # Resize to the input size of the model
        transforms.ToTensor(),                     # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=pixel_mean, std=pixel_std)  # Normalize using the dataset mean and std
    ])

    return fund_norm_transforms
    raise NotImplementedError(
        "`get_fundamental_normalization_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set,
    along with normalization. This should just be your previous method + normalization.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    all_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    all_transforms = transforms.Compose([
        transforms.Resize(inp_size),          # Resize to the specified input size
        
        # Apply color jitter to adjust brightness, contrast, saturation, and hue
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Randomly flip the image horizontally
        transforms.RandomHorizontalFlip(),
    

        transforms.ToTensor(),                # Convert the image to a PyTorch tensor
     
        transforms.Normalize(mean = pixel_mean, std= pixel_std), # Adjust mean and std as needed
    ])

    return all_transforms
    raise NotImplementedError(
        "`get_all_transforms` function in "
        + "`data_transforms.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    
