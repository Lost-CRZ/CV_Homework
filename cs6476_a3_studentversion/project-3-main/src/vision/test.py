import cv2
import numpy as np
from src.vision.part3_ransac import ransac_fundamental
import matplotlib.pyplot as plt
from src.vision.part3_ransac import compute_ransac_iterations

def panorama_stitch(imageA, imageB):
    # Convert images to grayscale
    imageA_color = cv2.imread(imageA)[:, :, ::-1]
    imageA_gray = cv2.cvtColor(imageA_color, cv2.COLOR_BGR2GRAY)
    
    imageB_color = cv2.imread(imageB)[:, :, ::-1]
    imageB_gray = cv2.cvtColor(imageB_color, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect keypoints and descriptors using SIFT (or any other feature detector)
    sift = cv2.SIFT_create()
    keypointsA, descriptorsA = sift.detectAndCompute(imageA_gray, None)
    keypointsB, descriptorsB = sift.detectAndCompute(imageB_gray, None)

    # Step 2: Match keypoints between images using Brute-Force Matcher and Ratio Test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptorsA, descriptorsB, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Step 3: Extract location of matched keypoints
    ptsA = np.float32([keypointsA[m.queryIdx].pt for m in good_matches])
    ptsB = np.float32([keypointsB[m.trainIdx].pt for m in good_matches])

    # Step 4: Compute homography using RANSAC
    H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC)

    # Step 5: Prepare panorama canvas and manually warp imageB into imageA's space
    A, B, __ = np.shape(imageA_color)
    M, N, __ = np.shape(imageB_color)
    
    Image_Height = max(A, M)
    Image_Width = int(B + N)  # Increase the canvas size
    panorama = np.zeros((Image_Height, Image_Width, 3), dtype=np.uint8)

    # Copy imageA to panorama
    panorama[0:A, 0:B, :] = imageA_color

    # Warp imageB manually using the homography matrix
    for i in range(M):
        for j in range(N):
            homogenious_coordB = np.array([j, i, 1]).reshape(3, 1)  # Correct homogeneous coordinates
            transformed_coord = np.dot(H, homogenious_coordB)
            x_B, y_B = int(transformed_coord[0]/transformed_coord[2]), int(transformed_coord[1]/transformed_coord[2])

            if 0 <= x_B < (Image_Width-1) and 0 <= y_B < (Image_Height-1):
                panorama[y_B, x_B, :] = imageB_color[i, j, :]

   
    origin=np.hstack((imageA_color,imageB_color))

    return panorama,origin


def test_transfer(imageA,imageB):

    imageA = cv2.imread(imageA)[:, :, ::-1]
    imageB = cv2.imread(imageB)[:, :, ::-1]

    # Step 1: Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypointsA, descriptorsA = sift.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = sift.detectAndCompute(imageB, None)

    # Step 2: Match the descriptors using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptorsA, descriptorsB)
   
    # Sort matches based on distance (i.e., quality of the match)
    matches = sorted(matches, key=lambda x: x.distance)
   
    # Step 3: Extract matched points from the matches
    ptsA = np.array([keypointsA[m.queryIdx].pt for m in matches], dtype=np.float32)
    ptsB = np.array([keypointsB[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Step 4: Estimate the fundamental matrix using RANSAC


    # Step 4: Estimate the homography matrix using RANSAC

    
    H1,__=cv2.findHomography(ptsB, ptsA, method=cv2.RANSAC)


    print(H1)

    # Get dimensions of both images
    heightA, widthA = imageA.shape[:2]
    heightB, widthB = imageB.shape[:2]

    # Initialize the panorama image
    panorama_width = widthA + widthB
    panorama_height = heightA  # Ensure output is A x (B + N)
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # Place imageA on the left side of the panorama
    panorama[0:heightA, 0:widthA] = imageA

    # Warp imageB manually and blend it onto the panorama
    for y in range(heightB):
        for x in range(widthB):
            # Get the homogenous coordinates
            new_coords = np.dot(H1, [x, y, 1])
            new_x = int(new_coords[0] / new_coords[2])  # Normalize by the third coordinate
            new_y = int(new_coords[1] / new_coords[2])

            # Check if the new coordinates fall within the bounds of the panorama
            if 0 <= new_x < panorama.shape[1] and 0 <= new_y < panorama.shape[0]:
                panorama[new_y, new_x,:] = imageB[y, x,:]

    compare=np.hstack((imageA,imageB))
    
    return panorama,compare


def visualize_matches(imageA_path, imageB_path):
        # 读取并转换图像 (BGR -> RGB)
    imageA = cv2.imread(imageA_path)[:, :, ::-1]
    imageB = cv2.imread(imageB_path)[:, :, ::-1]

    # Step 1: 使用 SIFT 检测关键点和描述符
    sift = cv2.SIFT_create()
    keypointsA, descriptorsA = sift.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = sift.detectAndCompute(imageB, None)

    # Step 2: 使用暴力匹配器 (Brute-Force Matcher) 匹配描述符
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptorsA, descriptorsB)
   
    # 按照匹配距离 (匹配质量) 排序
    matches = sorted(matches, key=lambda x: x.distance)
   
    # Step 3: 从匹配结果中提取匹配的点
    ptsA = np.array([keypointsA[m.queryIdx].pt for m in matches], dtype=np.float32)
    ptsB = np.array([keypointsB[m.trainIdx].pt for m in matches], dtype=np.float32)
    H, inliers_ptsA, inliers_ptsB = ransac_fundamental(ptsA, ptsB)
    # 拼接两张图片在一起
    stitched_image = np.hstack((imageA, imageB))

    # 可视化所有匹配点
    plt.figure(figsize=(12, 6))
    plt.imshow(stitched_image)
   
    width_A = imageA.shape[1]

    # 在拼接图像上绘制所有匹配点（黄色）
    for i in range(len(ptsA)):
        ptA = ptsA[i]
        ptB = ptsB[i] + np.array([width_A, 0])  # 将图像 B 的点平移到拼接图的位置
        # 绘制连接线
        plt.plot([ptA[0], ptB[0]], [ptA[1], ptB[1]], 'g-', linewidth=0.8)
        # 绘制匹配点
        plt.scatter([ptA[0], ptB[0]], [ptA[1], ptB[1]], s=10, color='yellow')

    # 如果有 inliers 提供，绘制它们的连接线和点（红色）
    if inliers_ptsA is not None and inliers_ptsB is not None:
        for i in range(len(inliers_ptsA)):
            ptA = inliers_ptsA[i]
            ptB = inliers_ptsB[i] + np.array([width_A, 0])  # 将图像 B 的点平移到拼接图的位置
            # 绘制内点连接线（红色）
            plt.plot([ptA[0], ptB[0]], [ptA[1], ptB[1]], 'r-', linewidth=1.5)
            # 绘制内点（红色）
            plt.scatter([ptA[0], ptB[0]], [ptA[1], ptB[1]], s=15, color='red')

    plt.axis('off')
    plt.show()