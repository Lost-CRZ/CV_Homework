#!/usr/bin/python3

import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    n1=features1.shape[0]
    n2=features2.shape[0]

    dists=np.zeros([n1,n2])
    
    for i in range(n1):
        #Substract i-th feature coord in feature1, from all feature coord in feature2
        #Gain the distance vector from ith point1 in feature to all points in feature2
        dists_ith_diff=features2-features1[i,:]
        #Gain the square of each coord in each distance vector
        dists_ith_sqr=dists_ith_diff*dists_ith_diff
        #Sum the each square value of coord in each distance vector
        dists_ith=np.sum(dists_ith_sqr,axis=1)
        #Gain the norm value of each distance vector
        dists_ith=np.sqrt(dists_ith)

        dists[i,:]=dists_ith        
        

    return dists
    raise NotImplementedError('`compute_feature_distances` function in ' +
        '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    n1,__=np.shape(features1)
    dists = compute_feature_distances(features1, features2)

    matches = []
    confidences = []

    for i in range(n1):
        # Sort the distance for each point in feature1, return their index in feature2 in increasing order
        sorted_indices = np.argsort(dists[i, :])
        best_match = sorted_indices[0]
        second_best_match = sorted_indices[1]

        # Compute the ratio of the distances
        ratio = dists[i, best_match] / dists[i, second_best_match]

        if ratio < 0.79:  # If ratio is less than threshold, the match reserves, otherwise, be discarded
            matches.append([i, best_match])
            confidences.append(1.0 - ratio)  

    # Convert matches and confidences to numpy arrays
    matches = np.array(matches)
    confidences = np.array(confidences)

    return matches, confidences
    raise NotImplementedError('`match_features_ratio_test` function in ' +
        '`part3_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    
