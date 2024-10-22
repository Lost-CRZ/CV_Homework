#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # Number of keypoints
    num=len(X)
    
    # Initialize feature vectors with zeros
    fvs=np.zeros((num, feature_width * feature_width))
    
    # Half width of the feature window
    half_width=feature_width//2
    
    for i in range(num):
        
        x,y=int(X[i]), int(Y[i])#Location of feature point
        
        #The top left corner
        x_origin=x-half_width+1 
        y_origin=y-half_width+1
        
        # Path range in image
        patch = image_bw[y_origin:y_origin + feature_width, x_origin:x_origin + feature_width]
        
        # Flatten the patch into a feature vector, namely convert the instense of the patch into one vector
        fv_point = patch.flatten()
        
        # Normalize the feature vector to have unit norm
        fv_point=fv_point/np.linalg.norm(fv_point)
        
        #feature vector at i-th feature poing
        fvs[i,:] = fv_point
    
    return fvs

    raise NotImplementedError('`compute_normalized_patch_descriptors` ' +
        'function in`part2_patch_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
