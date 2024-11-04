import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    num_pixels = 0

   # Use os.walk to traverse through directories
    for dirpath, dirnames, filenames in os.walk(dir_name):
        
        # Check if current directory has no subdirectories (deepest level)
        if not dirnames:
            # Process images in the current directory
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                # Check if the path is a file
                if os.path.isfile(file_path):

                    # Open the image and convert it to grayscale
                    with Image.open(file_path) as img:
                        img = img.convert("L")  # Convert to grayscale

                        # Convert image to numpy array and scale to [0, 1]
                        img_array = np.array(img) / 255.0

                        # Accumulate the sum and squared sum of pixel values
                        pixel_sum += img_array.sum()
                        pixel_squared_sum += (img_array ** 2).sum()
                        num_pixels += img_array.size

                       

    # Calculate mean and standard deviation
    if num_pixels > 0:
        mean = pixel_sum / num_pixels
        variance = (pixel_squared_sum / num_pixels) - (mean ** 2)
        std = np.sqrt(variance)
    else:
        raise ValueError("No valid images found in the directory.")

    return mean, std

    raise NotImplementedError(
            "`compute_mean_and_std` function in "
            + "`stats_helper.py` needs to be implemented"
        )

    ############################################################################
    # Student code end
    ############################################################################
    
