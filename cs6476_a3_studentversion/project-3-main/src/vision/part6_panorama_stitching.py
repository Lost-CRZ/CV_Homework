import numpy as np
import cv2 as cv

def panorama_stitch(imageA, imageB):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. This can be your own image pair that you believe
    will give you the best stitched panorama. Feel free to play around with 
    different image pairs as a fun exercise!

    In this task, you are encouraged to explore different feature detectors 
    and matchers (e.g., SIFT, SURF, ORB, Brute-Force, FLANN) and experiment 
    to see how different techniques affect the quality of the stitched panorama.
    
    You will:
    - Detect interest points in the two images using different feature detectors.
    - Match the interest points using various feature matchers.
    - Use the matched points to compute the homography matrix.
    - Warp one of the images into the coordinate space of the other image 
      manually to create a stitched panorama (note: you may NOT use any 
      pre-existing warping function like `warpPerspective`).

    The goal is to explore how the choice of feature detectors and matchers 
    influences the final panorama quality.

    Please note that you can use your fundamental matrix estimation from part3
    (imported for you above) to compute the homography matrix that you will 
    need to stitch the panorama.
    
    Feel free to reuse your interest point pipeline from project 2, or you may
    choose to use any existing interest point/feature matching functions from
    OpenCV. You may NOT use any pre-existing warping function though.

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using manual warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    # You are encouraged to explore different feature detectors and matchers. #
    # Experiment with different techniques to find what works best for your   #
    # chosen image pair, and use them to compute the homography matrix.       #
    # Remember: You may NOT use any pre-existing warping functions!           #
    ###########################################################################

    # Load and convert images
    imageA_color = cv.imread(imageA)[:, :, ::-1]
    imageA_gray = cv.cvtColor(imageA_color, cv.COLOR_BGR2GRAY)
    
    imageB_color = cv.imread(imageB)[:, :, ::-1]
    imageB_gray = cv.cvtColor(imageB_color, cv.COLOR_BGR2GRAY)

    choice="SIFT"

    if choice=="SIFT":
        #Detect keypoints and descriptors using SIFT 
        sift = cv.SIFT_create()
        keypointsA, descriptorsA = sift.detectAndCompute(imageA_gray, None)
        keypointsB, descriptorsB = sift.detectAndCompute(imageB_gray, None)


        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptorsA, descriptorsB, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        # Compute the corresponding point pairs and the homography matrix
        ptsA = np.float32([keypointsA[m.queryIdx].pt for m in good_matches])
        ptsB = np.float32([keypointsB[m.trainIdx].pt for m in good_matches])

    
    if choice=="ORB":
        #Using ORB to find feature points
        orb = cv.ORB_create()
        keypointsA, descriptorsA = orb.detectAndCompute(imageA_gray, None)
        keypointsB, descriptorsB = orb.detectAndCompute(imageB_gray, None)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  
        matches = bf.match(descriptorsA, descriptorsB)
    
        matches = sorted(matches, key=lambda x: x.distance)

        ptsA = np.float32([keypointsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([keypointsB[m.trainIdx].pt for m in matches])



    Homography_matrix, status = cv.findHomography(ptsB, ptsA, cv.RANSAC)   

    # Image sizes
    A, B, __ = np.shape(imageA_color)
    M, N, __ = np.shape(imageB_color)
    print(f"A,B={A,B} M,N={M,N}")
    
    Image_Height = max(A, M)
    Image_Width = B + N
    panorama = np.zeros((Image_Height, Image_Width, 3), dtype=np.uint8)
    panorama[0:A,0:B,:]=imageA_color

    # Transform points from imageB to imageA one by one
    overlapped_B_in_A_x = []
    overlapped_B_in_A_y = []
    overlapped_coord_B_x = []
    overlapped_coord_B_y = []

    for i in range(M):
        for j in range(N):
            homogenious_coordB = np.array([[j], [i], [1]])  # x, y coordinates in homogeneous form
            transfered_homogeious_coordB = Homography_matrix @ homogenious_coordB  # Apply homography
            transfered_coordB = transfered_homogeious_coordB[0:2] / transfered_homogeious_coordB[2]  # x/s, y/s
            x_B, y_B = np.round(transfered_coordB[0]).astype(int), np.round(transfered_coordB[1]).astype(int)

            if (x_B >= 0) and (x_B < Image_Width) and (y_B >= 0) and (y_B < Image_Height):
                if (x_B < B) and (y_B < A):  # Overlapped region
                    overlapped_B_in_A_x.append(x_B)
                    overlapped_B_in_A_y.append(y_B)
                    overlapped_coord_B_x.append(i)
                    overlapped_coord_B_y.append(j)
                else:
                    panorama[y_B, x_B, :] = imageB_color[i, j, :]

    # Blend the overlapped region
    print(overlapped_B_in_A_x)
    if overlapped_B_in_A_x:
        overlapped_B_in_A_x = np.array(overlapped_B_in_A_x)
        overlapped_B_in_A_y = np.array(overlapped_B_in_A_y)
        overlapped_coord_B_x = np.array(overlapped_coord_B_x)
        overlapped_coord_B_y = np.array(overlapped_coord_B_y)

        x_start = min(overlapped_B_in_A_x)
        x_end = max(overlapped_B_in_A_x)

        for i in range(np.shape(overlapped_B_in_A_x)[0]):
            coeffi = (overlapped_B_in_A_x[i] - x_start) / (x_end - x_start)
            # Ensure indices are within bounds
            x_clamped = min(overlapped_B_in_A_x[i], B - 1)
            y_clamped = min(overlapped_B_in_A_y[i], A - 1)

            if x_clamped>=1200:
                continue
              
            panorama[y_clamped, x_clamped, :] = (
                (1-coeffi) * imageA_color[y_clamped, x_clamped, :] +
                coeffi * imageB_color[overlapped_coord_B_x[i], overlapped_coord_B_y[i], :]
            )

    
    return panorama
    # raise NotImplementedError("`panorama_stitch` function in "
    #     + "`part6_panorama_stitching.py` needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    