#!/usr/bin/python3

import copy
import pdb
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vision.part1_harris_corner import get_harris_interest_points
from src.vision.part3_feature_matching import match_features_ratio_test
from src.vision.part4_sift_descriptor import (
    get_orientations_and_magnitudes,
    get_gradient_histogram_vec_from_patch,
    get_sift_features_vectorized,
    get_SIFT_descriptors,
    get_feat_vec,
)
from src.vision.utils import load_image, evaluate_correspondence, rgb2gray, PIL_resize

ROOT = Path(__file__).resolve().parent.parent  # ../..


def test_get_orientations_and_magnitudes():
    """ Verify gradient orientations and magnitudes are computed correctly"""
    Ix = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Iy = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    orientations, magnitudes = get_orientations_and_magnitudes(Ix, Iy)

    # there are 3 vectors -- (1,0) at 0 deg, (0,1) at 90 deg, and (-1,1) and 135 deg
    expected_orientations = np.array(
        [[0, 0, 0], [np.pi / 2, np.pi / 2, np.pi / 2], [3 * np.pi / 4, 3 * np.pi / 4, 3 * np.pi / 4]]
    )
    expected_magnitudes = np.array([[1, 1, 1], [1, 1, 1], [np.sqrt(2), np.sqrt(2), np.sqrt(2)]])

    assert np.allclose(orientations, expected_orientations)
    assert np.allclose(magnitudes, expected_magnitudes)


def test_get_gradient_histogram_vec_from_patch():
    """ Check if weighted gradient histogram is computed correctly """
    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]

    window_orientations = np.array(
        [
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]
    )

    window_magnitudes = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    wgh = get_gradient_histogram_vec_from_patch(window_orientations, window_magnitudes)

    expected_wgh = np.array(
        [
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4, magnitude 1
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4, magnitude 0
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.], # bin 5, magnitude 2
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.]
        ]
    ).reshape(128, 1)

    assert np.allclose(wgh, expected_wgh, atol=1e-1)


def test_get_feat_vec():
    """ Check if feature vector for a specific interest point is returned correctly """
    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]
    C = 5/8 * np.pi # squarely in bin [pi/2, 3pi/4]

    window_orientations = np.array(
        [
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ]
        ]
    )

    window_magnitudes = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    feature_width = 16

    x, y = 7, 8

    fv = get_feat_vec(x, y, window_orientations, window_magnitudes, feature_width)

    expected_fv = np.array(
        [
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ]
        ]
    ).reshape(128, 1)

    assert np.allclose(fv, expected_fv, atol=1e-2)


def test_get_SIFT_descriptors():
    """ Check if the 128-d SIFT feature vector computed at each of the input points is returned correctly """

    image1 = np.array(
        [
            [27, 33, 27, 29, 35, 42, 46, 52, 58, 64, 59, 53, 59, 62, 56, 59, 63, 58, 53, 57],
            [32, 36, 31, 35, 31, 35, 38, 36, 43, 39, 33, 39, 35, 38, 36, 40, 36, 42, 45, 51],
            [33, 31, 30, 36, 39, 43, 38, 45, 43, 49, 43, 41, 47, 45, 51, 55, 58, 56, 62, 66],
            [35, 42, 45, 48, 43, 46, 50, 45, 42, 36, 30, 24, 31, 26, 24, 30, 34, 40, 47, 44],
            [31, 38, 44, 42, 47, 45, 50, 45, 39, 43, 50, 44, 39, 36, 40, 46, 53, 50, 47, 42],
            [30, 25, 19, 17, 15, 16, 21, 28, 33, 37, 38, 43, 47, 45, 51, 57, 62, 57, 63, 68],
            [33, 39, 43, 40, 44, 38, 39, 40, 38, 36, 42, 48, 54, 60, 57, 63, 57, 51, 45, 51],
            [29, 27, 33, 40, 47, 52, 58, 56, 58, 64, 58, 59, 63, 67, 61, 67, 72, 73, 74, 75],
            [28, 25, 32, 37, 42, 47, 41, 38, 32, 37, 32, 36, 31, 36, 33, 31, 28, 22, 16, 20],
            [31, 36, 34, 32, 26, 24, 31, 33, 30, 24, 19, 17, 19, 13,  7,  5,  3,  0,  2,  7],
            [32, 29, 34, 39, 33, 35, 37, 31, 38, 32, 38, 44, 51, 57, 54, 60, 58, 56, 50, 55],
            [36, 38, 45, 44, 49, 44, 43, 49, 44, 43, 45, 41, 35, 37, 39, 34, 28, 34, 41, 46],
            [40, 35, 34, 28, 24, 27, 33, 29, 23, 28, 31, 26, 31, 26, 31, 28, 27, 23, 22, 28],
            [38, 33, 35, 40, 46, 41, 47, 49, 44, 46, 43, 50, 57, 62, 67, 63, 69, 66, 73, 70],
            [42, 48, 54, 50, 45, 40, 46, 52, 48, 42, 44, 46, 45, 46, 52, 58, 65, 67, 69, 65],
            [45, 50, 46, 52, 57, 62, 61, 68, 63, 70, 69, 74, 81, 87, 93, 98, 92, 91, 92, 98],
            [42, 43, 48, 54, 49, 43, 39, 45, 51, 52, 46, 51, 47, 41, 47, 43, 49, 44, 39, 33],
            [44, 40, 41, 37, 33, 27, 34, 29, 25, 30, 35, 31, 32, 38, 45, 50, 44, 51, 45, 41],
            [42, 36, 41, 44, 38, 34, 37, 32, 35, 41, 42, 45, 52, 56, 61, 62, 67, 72, 68, 71],
            [41, 44, 45, 49, 50, 57, 64, 59, 63, 66, 62, 56, 63, 58, 65, 68, 62, 63, 64, 58],
        ]
    ).astype(np.float32)

    X1, Y1 = np.array([9, 10]), np.array([9, 10])

    SIFT_descriptors = get_SIFT_descriptors(image1, X1, Y1)

    expected_SIFT_descriptors = np.array(
        [
            [
                [0.074, 0.091, 0.400, 0.000, 0.000, 0.335, 0.000, 0.000],
                [0.152, 0.191, 0.270, 0.084, 0.096, 0.160, 0.102, 0.091],
                [0.109, 0.114, 0.218, 0.105, 0.096, 0.272, 0.245, 0.113],
                [0.000, 0.000, 0.233, 0.230, 0.000, 0.400, 0.182, 0.062],
                [0.066, 0.106, 0.303, 0.216, 0.151, 0.374, 0.000, 0.000],
                [0.096, 0.408, 0.111, 0.086, 0.000, 0.396, 0.000, 0.088],
                [0.000, 0.405, 0.405, 0.000, 0.000, 0.345, 0.243, 0.000],
                [0.000, 0.673, 0.201, 0.000, 0.000, 0.177, 0.436, 0.000],
                [0.000, 0.213, 0.106, 0.111, 0.000, 0.338, 0.268, 0.000],
                [0.000, 0.146, 0.127, 0.000, 0.111, 0.286, 0.324, 0.109],
                [0.000, 0.225, 0.218, 0.000, 0.000, 0.540, 0.042, 0.000],
                [0.000, 0.342, 0.206, 0.000, 0.000, 0.508, 0.499, 0.000],
                [0.000, 0.347, 0.225, 0.000, 0.087, 0.220, 0.183, 0.000],
                [0.000, 0.259, 0.439, 0.104, 0.000, 0.250, 0.242, 0.000],
                [0.000, 0.252, 0.427, 0.112, 0.141, 0.414, 0.000, 0.000],
                [0.000, 0.414, 0.381, 0.124, 0.000, 0.469, 0.186, 0.000],
            ],
            [
                [0.000, 0.087, 0.404, 0.000, 0.000, 0.420, 0.097, 0.000],
                [0.124, 0.182, 0.139, 0.128, 0.082, 0.352, 0.103, 0.139],
                [0.000, 0.109, 0.148, 0.000, 0.092, 0.459, 0.210, 0.000],
                [0.000, 0.000, 0.136, 0.200, 0.081, 0.339, 0.293, 0.059],
                [0.063, 0.206, 0.304, 0.158, 0.098, 0.209, 0.205, 0.000],
                [0.091, 0.422, 0.055, 0.083, 0.000, 0.219, 0.219, 0.084],
                [0.000, 0.435, 0.387, 0.000, 0.000, 0.307, 0.335, 0.000],
                [0.000, 0.562, 0.398, 0.000, 0.000, 0.296, 0.469, 0.000],
                [0.000, 0.177, 0.129, 0.078, 0.071, 0.330, 0.233, 0.000],
                [0.000, 0.139, 0.137, 0.000, 0.079, 0.299, 0.330, 0.104],
                [0.000, 0.279, 0.182, 0.000, 0.000, 0.569, 0.000, 0.000],
                [0.000, 0.275, 0.274, 0.000, 0.000, 0.626, 0.287, 0.000],
                [0.000, 0.331, 0.249, 0.099, 0.083, 0.201, 0.272, 0.000],
                [0.000, 0.247, 0.429, 0.000, 0.000, 0.373, 0.195, 0.000],
                [0.000, 0.240, 0.416, 0.160, 0.135, 0.413, 0.000, 0.000],
                [0.000, 0.464, 0.331, 0.000, 0.000, 0.328, 0.302, 0.000],
            ],
        ]
    ).reshape(2, 128)

    assert np.allclose(SIFT_descriptors, expected_SIFT_descriptors, atol=1e-1)


def test_feature_matching_speed():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must run in under 90 seconds.
    """
    start = time.time()
    image1 = load_image(f"{ROOT}/data/1a_notredame.jpg")
    image2 = load_image(f"{ROOT}/data/1b_notredame.jpg")
    eval_file = f"{ROOT}/ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(copy.deepcopy(image1_bw))
    X2, Y2, _ = get_harris_interest_points(copy.deepcopy(image2_bw))

    image1_features = get_SIFT_descriptors(image1_bw, X1, Y1)
    image2_features = get_SIFT_descriptors(image2_bw, X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)
    print("{:d} matches from {:d} corners".format(len(matches), len(X1)))

    end = time.time()
    duration = end - start
    print(f"Your Feature matching pipeline takes {duration:.2f} seconds to run on Notre Dame")

    MAX_ALLOWED_TIME = 90  # sec
    assert duration < MAX_ALLOWED_TIME


def test_feature_matching_accuracy():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must achieve at least 80% accuracy.
    """
    image1 = load_image(f"{ROOT}/data/1a_notredame.jpg")
    image2 = load_image(f"{ROOT}/data/1b_notredame.jpg")
    eval_file = f"{ROOT}/ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(copy.deepcopy(image1_bw))
    X2, Y2, _ = get_harris_interest_points(copy.deepcopy(image2_bw))

    image1_features = get_SIFT_descriptors(image1_bw, X1, Y1)
    image2_features = get_SIFT_descriptors(image2_bw, X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)

    acc, _ = evaluate_correspondence(
        image1,
        image2,
        eval_file,
        scale_factor,
        X1[matches[:, 0]],
        Y1[matches[:, 0]],
        X2[matches[:, 1]],
        Y2[matches[:, 1]],
    )

    print(f"Your Feature matching pipeline achieved {100 * acc:.2f}% accuracy to run on Notre Dame")

    MIN_ALLOWED_ACC = 0.80  # 80 percent
    assert acc > MIN_ALLOWED_ACC


def test_extra_credit_vectorized_sift():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must run in under 90 seconds.
    """

    image1 = load_image(f"{ROOT}/data/1a_notredame.jpg")
    image2 = load_image(f"{ROOT}/data/1b_notredame.jpg")
    eval_file = f"{ROOT}/ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(copy.deepcopy(image1_bw))
    X2, Y2, _ = get_harris_interest_points(copy.deepcopy(image2_bw))

    start = time.time()
    image1_features = get_sift_features_vectorized(image1_bw, X1, Y1)
    image2_features = get_sift_features_vectorized(image2_bw, X2, Y2)
    end = time.time()
    duration = end - start
    print(f"Your vectorized SIFT implementation takes {duration:.2f} seconds to run on Notre Dame")

    MAX_ALLOWED_TIME = 5  # sec
    assert duration < MAX_ALLOWED_TIME, "Runtime too long"

    matches, confidences = match_features_ratio_test(image1_features, image2_features)
    print("{:d} matches from {:d} corners".format(len(matches), len(X1)))

    acc, _ = evaluate_correspondence(
        image1,
        image2,
        eval_file,
        scale_factor,
        X1[matches[:, 0]],
        Y1[matches[:, 0]],
        X2[matches[:, 1]],
        Y2[matches[:, 1]],
    )
    print(f"Your vectorized feature matching pipeline achieved {100 * acc:.2f}% accuracy to run on Notre Dame")

    MIN_ALLOWED_ACC = 0.80  # 80 percent
    assert acc > MIN_ALLOWED_ACC, "Accuracy too low"
