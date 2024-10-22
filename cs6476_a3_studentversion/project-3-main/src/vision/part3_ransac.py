import math
import numpy as np
from vision.part2_fundamental_matrix import compute_fundamental_matrix,standardize_points,denormalize_fundamental_matrix

def compute_ransac_iterations(
    success_prob: float, sample_count: int, inlier_prob: float
) -> int:
    """
    Computes the number of RANSAC iterations needed to achieve a certain success probability.

    Args:
        success_prob: Desired probability of successful estimation (all in line)
        sample_count: Number of points sampled in each iteration
        inlier_prob: Probability that a single point is an inlier

    Returns:
        num_iterations: Number of RANSAC iterations required
    """
    num_iterations = None
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################

     # Calculate the failure probability for a single iteration
    failure_prob = 1 - inlier_prob**sample_count

    # Calculate the number of iterations needed to achieve the desired success probability
    num_iterations = np.log(1 - success_prob) / np.log(failure_prob)

    return int(np.ceil(num_iterations))
    raise NotImplementedError("`compute_ransac_iterations` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    

def ransac_fundamental(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Uses RANSAC to estimate the fundamental matrix between two sets of points.

    Tips:
        - Determine appropriate values for success_prob, sample_count, and inlier_prob.
        - Use numpy.random.choice to select random samples.
        - An error threshold of 0.1 is recommended for distinguishing inliers.

    Args:
        pts_a: An array of shape (N, 2) containing points from image A
        pts_b: An array of shape (N, 2) containing points from image B

    Returns:
        best_fundamental_matrix: The estimated fundamental matrix
        inlier_pts_a: Inlier points from image A
        inlier_pts_b: Inlier points from image B
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################

    success_prob = 0.95  # Probability of RANSAC succeeding
    sample_count = 8     # Minimum number of points for estimating fundamental matrix
    inlier_prob = 0.7    # Estimated probability of any single point being an inlier
    error_threshold = 0.1  # Error threshold to consider points as inliers

    # Compute number of RANSAC iterations
    iteration_num = compute_ransac_iterations(success_prob, sample_count, inlier_prob)

    best_fundamental_matrix = None
    inlier_pts_a = None
    inlier_pts_b = None
    

    N = pts_a.shape[0] #Number of corresponding pairs

    best_inliers_count=0
    #i-th sample from dataset to calculate foundamental matrix
    for i in range(iteration_num):
        
        inliers_count = 0
        #select sample_count num from dataset for calculating the matrix
        selected_index=np.random.choice(N,sample_count,replace=False)#generate sample_count random int number from 0 to N, with no repeat num
        pts_a_selected=pts_a[selected_index]
        pts_b_selected=pts_b[selected_index]

        #calculate the foundamental matrix based on selected points
        #pts_a_norm,Ta=standardize_points(pts_a_selected)
        #pts_b_norm,Tb=standardize_points(pts_b_selected)

        #F_norm=compute_fundamental_matrix(pts_a_norm,pts_b_norm)

        #F=denormalize_fundamental_matrix(F_norm,Ta,Tb)
        F=compute_fundamental_matrix(pts_a_selected,pts_b_selected)

        in_liers_index=[]
        #Test the quality of calculated foundamental matrix that is calculated based on all sample correspondings
        for j in range(N):
            #calculate the epipoline for j-th pair in dataset
            
            #gain the line parameters of i-th pts a in image b aX+bY+c=0, F@[pts_a_x,pts_a,y,1]=l_b=[a,b,c]
            #the fundamental matrix will convert a point in one image into an epipoline in the another image
            expanded_a=np.append(pts_a[j],1)
            expanded_b=np.append(pts_b[j],1)

            epi_l_b=F@expanded_a
            epi_l_a=F.T@expanded_b

            #square distance from point b to epipoine line b, representing the error
            dis_b2lb=(expanded_b@epi_l_b)**2/(epi_l_b[0]**2+epi_l_b[1]**2)

            dis_a2la=(expanded_a@epi_l_a)**2/(epi_l_a[0]**2+epi_l_a[1]**2)

            #judge whether this point is in lines
            if (dis_a2la+dis_b2lb)<error_threshold:
                inliers_count+=1
                in_liers_index.append(j) #record the index for the corresponding pair in the sequence of selected points
        
        in_liers_index=np.array(in_liers_index)
      
        #find the best matrix based on how many points in all dataset are in liers
        if inliers_count>best_inliers_count:
            best_inliers_count=inliers_count
            best_fundamental_matrix=F
            inlier_pts_a=pts_a[in_liers_index]
            inlier_pts_b=pts_b[in_liers_index]


    

    return best_fundamental_matrix, inlier_pts_a, inlier_pts_b
    raise NotImplementedError("`ransac_fundamental` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

   
