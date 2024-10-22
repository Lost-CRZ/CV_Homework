#!/usr/bin/python3

import copy
from pathlib import Path

import numpy as np
from vision.part1 import (
    create_Gaussian_kernel_1D,
    create_Gaussian_kernel_2D,
    create_hybrid_image,
    my_conv2d_numpy,
)
from vision.utils import (
    load_image,
)

ROOT = Path(__file__).resolve().parent.parent  # ../..

"""
Even size kernels are not required for this project, so we exclude this test case.
"""


def test_create_Gaussian_kernel_1D():
    """Check that a few values are correct inside 1d kernel"""
    ksize = 25
    sigma = 7
    kernel = create_Gaussian_kernel_1D(ksize, sigma)

    ksize_2 = 29
    sigma_2 = 7
    kernel_2 = create_Gaussian_kernel_1D(ksize_2, sigma_2)

    assert kernel.shape == (25, 1), "The kernel is not the correct size"
    assert kernel_2.shape == (29, 1), "The kernel is not the correct size"
    kernel = kernel.squeeze()
    kernel_2 = kernel_2.squeeze()

    # peak should be at center
    gt_kernel_crop = np.array(
        [0.05614, 0.05908, 0.06091, 0.06154, 0.06091, 0.05908, 0.05614]
    )
    gt_kernel_crop_2 = np.array(
        [0.05405, 0.05688, 0.05865, 0.05925, 0.05865, 0.05688, 0.05405]
    )
    h_center = ksize // 2
    h_center_2 = ksize_2 // 2
    student_kernel_crop = kernel[h_center - 3 : h_center + 4]
    student_kernel_crop_2 = kernel_2[h_center_2 - 3 : h_center_2 + 4]

    assert np.allclose(
        gt_kernel_crop, student_kernel_crop, atol=1e-5
    ), "Values dont match"
    assert np.allclose(
        gt_kernel_crop_2, student_kernel_crop_2, atol=1e-5
    ), "Values dont match"
    assert np.allclose(kernel.sum(), 1, atol=1e-3), "Kernel doesnt sum to 1"
    assert np.allclose(kernel_2.sum(), 1, atol=1e-3), "Kernel doesnt sum to 1"


def test_create_Gaussian_kernel_1D_sumsto1():
    """Verifies that generated 1d Gaussian kernel sums to 1."""
    ksize = 25
    sigma = 7
    kernel = create_Gaussian_kernel_1D(ksize, sigma)
    assert np.allclose(kernel.sum(), 1, atol=1e-3), "Kernel doesnt sum to 1"


def test_create_Gaussian_kernel_2D_sumsto1():
    """Verifies that generated 2d Gaussian kernel sums to 1."""
    cutoff_frequency = 5
    kernel = create_Gaussian_kernel_2D(cutoff_frequency)
    assert np.allclose(kernel.sum(), 1, atol=1e-3), "Kernel doesnt sum to 1"


def test_create_Gaussian_kernel_1D_peak():
    """Ensure peak of 1d kernel is at center, and dims are correct"""
    ksize = 25
    sigma = 7
    kernel = create_Gaussian_kernel_1D(ksize, sigma)

    # generated Gaussian kernel should have odd dimensions
    assert kernel.shape[0] % 2 == 1, "Gaussian kernel should have odd dimensions"
    assert kernel.shape[1] % 2 == 1, "Gaussian kernel should have odd dimensions"
    assert kernel.ndim == 2

    center_idx = kernel.shape[0] // 2

    assert kernel.squeeze().argmax() == center_idx, "Peak is not at center index"

    coords = np.where(kernel == kernel.max())
    coords = np.array(coords).T

    # should be only 1 peak
    assert coords.shape == (1, 2), "Peak is not unique"


def test_create_Gaussian_kernel_2D_peak():
    """Ensure peak of 2d kernel is at center, and dims are correct"""
    cutoff_frequency = 5
    kernel = create_Gaussian_kernel_2D(cutoff_frequency)

    # generated Gaussian kernel should have odd dimensions
    assert kernel.shape[0] % 2 == 1
    assert kernel.shape[1] % 2 == 1
    assert kernel.ndim == 2

    center_row = kernel.shape[0] // 2
    center_col = kernel.shape[1] // 2

    coords = np.where(kernel == kernel.max())
    coords = np.array(coords).T

    # should be only 1 peak
    assert coords.shape == (1, 2), "Peak is not unique"
    assert coords[0, 0] == center_row, "Peak is not at center row"
    assert coords[0, 1] == center_col, "Peak is not at center column"


def test_gaussian_kernel_2D() -> None:
    """Verify values of inner 5x5 patch of 21x21 Gaussian kernel."""
    cutoff_frequency = 5
    kernel = create_Gaussian_kernel_2D(cutoff_frequency)
    assert kernel.shape == (21, 21), "The kernel is not the correct size"

    # peak should be at center
    gt_kernel_crop = np.array(
        [
            [0.00583, 0.00619, 0.00632, 0.00619, 0.00583],
            [0.00619, 0.00657, 0.00671, 0.00657, 0.00619],
            [0.00632, 0.00671, 0.00684, 0.00671, 0.00632],
            [0.00619, 0.00657, 0.00671, 0.00657, 0.00619],
            [0.00583, 0.00619, 0.00632, 0.00619, 0.00583],
        ]
    )

    kernel_h, kernel_w = kernel.shape
    h_center = kernel_h // 2
    w_center = kernel_w // 2
    student_kernel_crop = kernel[
        h_center - 2 : h_center + 3, w_center - 2 : w_center + 3
    ]

    assert np.allclose(
        gt_kernel_crop, student_kernel_crop, atol=1e-5
    ), "Values dont match"
    assert np.allclose(kernel.sum(), 1.0, atol=1e-3)


def test_my_conv2d_numpy_identity()->np.array:
    """Check identity filter works correctly on all channels"""
    filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    image_path = 'data/4a_einstein.bmp'
    filtered_img = my_conv2d_numpy(image_path, filter)

    #return filtered_img
    assert np.allclose(filtered_img, load_image(image_path))


def test_my_conv2d_numpy_ones_filter():
    """Square filter of all 1s"""
    filter = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    image_path = 'data/test1.bmp'

    filtered_img = my_conv2d_numpy(image_path, filter)

    img = load_image(image_path)
    s = np.sum(img[:,:,0])
    gt_filtered_channel_img = np.array([[s, s], [s, s]])
    gt_filtered_img = np.zeros((2, 2, 3), dtype=np.float32)
    for i in range(3):
        gt_filtered_img[:, :, i] = gt_filtered_channel_img
    
    assert np.allclose(filtered_img, gt_filtered_img)


def test_my_conv2d_numpy_nonsquare_filter():
    """ """

    filter = np.array([[1, 1, 1]])

    img = 'data/test2.bmp'

    filtered_img = my_conv2d_numpy(img, filter)

    gt_filtered_channel_img = np.array([[2, 3, 2], [2, 3, 2]])
    gt_filtered_img = np.zeros((2, 3, 3), dtype=np.uint8)
    for i in range(3):
        gt_filtered_img[:, :, i] = gt_filtered_channel_img

    assert np.allclose(filtered_img, gt_filtered_img)


def test_hybrid_image_np() -> None:
    """Verify that hybrid image values are correct."""
    image_path1 = f"{ROOT}/data/1a_dog.bmp"
    image_path2 = f"{ROOT}/data/1b_cat.bmp"
    kernel = create_Gaussian_kernel_2D(7)
    _, _, hybrid_image = create_hybrid_image(image_path1, image_path2, kernel)

    img_h, img_w, _ = load_image(image_path2).shape
    k_h, k_w = kernel.shape
    # Exclude the border pixels.
    hybrid_interior = hybrid_image[k_h : img_h - k_h, k_w : img_w - k_w]
    correct_sum = np.allclose(158339.52, hybrid_interior.sum())

    # ground truth values
    gt_hybrid_crop = np.array(
        [
            [[0.5429589, 0.55373234, 0.5452099], [0.5290553, 0.5485607, 0.545738]],
            [[0.55020595, 0.55713284, 0.5457024], [0.5368045, 0.5603536, 0.5505791]],
        ],
        dtype=np.float32,
    )

    # H,W,C order in Numpy
    correct_crop = np.allclose(
        hybrid_image[100:102, 100:102, :], gt_hybrid_crop, atol=1e-3
    )
    assert (
        correct_sum and correct_crop
    ), "Hybrid image values are not correct, please double check your implementation."

    ## Purely for debugging/visualization ##
    # plt.imshow(hybrid_image)
    # plt.show()
    ########################################
