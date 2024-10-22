"""Fundamental matrix utilities."""

import numpy as np

def compute_fundamental_matrix(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> np.ndarray:
    """
    Estimates the fundamental matrix using point correspondences between two images.

    Args:
        pts_a: A numpy array of shape (N, 2) representing points in image A
        pts_b: A numpy array of shape (N, 2) representing points in image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    
    # Number of points
    N = pts_a.shape[0]
    
    # Initialize matrix A (2N x 9)
    A = np.zeros((N, 9))
    
    # Fill the matrix A with the known 2D and 3D point correspondences
    for i in range(N):
        u, v = pts_a[i]
        u_prime, v_prime = pts_b[i]

        A[i]=[u*u_prime,v*u_prime,u_prime,u*v_prime,v*v_prime,v_prime,u,v,1]
    
    # Solve for P using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 3)  # The last row of Vt is the solution for P

    # The solution is the last row of Vt (or last column of V)
    F_3 = Vt[-1].reshape(3, 3)
    
    # Enforce the rank-2 constraint by performing SVD on F
    U_F, S_F, Vt_F = np.linalg.svd(F_3)
    
    # Set the smallest singular value to zero
    S_F[2] = 0
    
    # Reconstruct the fundamental matrix with rank 2
    F = U_F @ np.diag(S_F) @ Vt_F
    
    return F
    raise NotImplementedError("`compute_fundamental_matrix` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################



def denormalize_fundamental_matrix(
    F_normalized: np.ndarray, T_a: np.ndarray, T_b: np.ndarray
) -> np.ndarray:
    """
    Adjusts the normalized fundamental matrix using the transformation matrices.

    Args:
        F_normalized: A numpy array of shape (3, 3) representing the normalized fundamental matrix
        T_a: Transformation matrix for image A
        T_b: Transformation matrix for image B

    Returns:
        F_denormalized: A numpy array of shape (3, 3) representing the original fundamental matrix
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################

    F_denormalized=T_b.T @ F_normalized @ T_a

    return F_denormalized
    raise NotImplementedError("`denormalize_fundamental_matrix` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    

def standardize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Normalizes 2D points to improve numerical stability in computations.

    Args:
        points: A numpy array of shape (N, 2) representing the 2D points

    Returns:
        points_standardized: A numpy array of shape (N, 2) representing the normalized 2D points
        T: The transformation matrix used for normalization
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################

    N=points.shape[0]

    points_x=points[:,0]
    points_y=points[:,1]

    center_x=np.sum(points_x)/N
    center_y=np.sum(points_y)/N

    scale_x=1/np.std(points_x-center_x)
    scale_y=1/np.std(points_y-center_y)

    scale_M=np.array(([scale_x,0,0],[0,scale_y,0],[0,0,1]))
    translation_M=np.array(([1,0,-center_x],[0,1,-center_y],[0,0,1]))

    added_matrix=np.ones((N,1))
    points_added=np.hstack((points,added_matrix)) #[N,3]

    T=scale_M@translation_M

    points_standardized=[]

    for i in range(N):
     points_standardized.append(T@points_added[i,:]) #for i-th point 

    points_standardized=np.array(points_standardized)[:,0:2]
    return points_standardized, T
    raise NotImplementedError("`standardize_points` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    
