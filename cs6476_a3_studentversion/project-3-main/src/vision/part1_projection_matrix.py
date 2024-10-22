import numpy as np

def compute_camera_center(P: np.ndarray) -> np.ndarray:
    """
    Computes the camera center location from a given projection matrix.

    Args:
        P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
        camera_center: A numpy array of shape (1, 3) representing the camera center
                       location in world coordinates
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################

    #Projection Matrix
    Q=P[:,0:3]
    #Translation Matrix
    t=P[:,3]
    #[Q t][x;y;z;1]=0--->C=-inv(Q)*t
    Q_inv=np.linalg.inv(Q)
    camera_center=-Q_inv@t # @:Matrix product, * : element-wise product

    return camera_center
    raise NotImplementedError("`compute_camera_center` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    

def project_points(M: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Projects 3D points into 2D image coordinates using the projection matrix.

    Args:
        M: A 3 x 4 numpy array representing the projection matrix
        points_3d: A numpy array of shape (N, 3) representing 3D points

    Returns:
        points_2d: A numpy array of shape (N, 2) representing projected 2D points
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################

    N=np.shape(points_3d)[0]#N points
    add_3d=np.ones((N,1))
    points_3d_added=np.hstack((points_3d,add_3d)) #[N,4]

    scale_factor=points_3d_added@np.transpose(M[2,:]) #[N,4]@[4,1]=[N], each row is the scalar for i-th point

    points_2d_added=[]
    for i in range(N):
        #M@np.transpose(points_3d_added[i,:])  [3,4]@transpose([1,4])=[3,1]--->[u*s;v*s;s] for i-th points
        #[3,1]/s (i-th scalar)-->[u;v;1]
        points_2d_added.append((M@np.transpose(points_3d_added[i,:])/scale_factor[i])) #append here actually create a new dimension at axis=0

    points_2d_added=np.array(points_2d_added)
    print(np.shape(points_2d_added))
    points_2d=points_2d_added[:,0:2]#[N,3]-->[N,2]
    return points_2d
    raise NotImplementedError("`project_points` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    

def compute_projection_matrix(image_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """
    Computes the projection matrix from corresponding 2D-3D point pairs.

    To compute the projection matrix, set up a system of equations using the known 2D and 3D point correspondences. You can then solve for the projection matrix using least squares or SVD. Note that each point pair provides two equations, and at least 6 point pairs are needed to solve for the projection matrix.

    Args:
        image_points: A numpy array of shape (N, 2)
        world_points: A numpy array of shape (N, 3)

    Returns:
        P: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################

    # Number of points
    N = image_points.shape[0]
    
    # Initialize matrix A (2N x 12)
    A = np.zeros((2 * N, 12))
    
    # Fill the matrix A with the known 2D and 3D point correspondences
    for i in range(N):
        X, Y, Z = world_points[i]
        u, v = image_points[i]
        
        A[2 * i]     = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]
    
    # Solve for P using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  # The last row of Vt is the solution for P

    return P
    raise NotImplementedError("`compute_projection_matrix` function needs to be implemented")

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    
