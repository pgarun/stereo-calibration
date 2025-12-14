import numpy as np

class StereoCalibration:
    def __init__(self, k1, k2, F):
        self.k1 = k1
        self.k2 = k2
        self.F = F
    
    def _essential_matrix(self):
        """Compute the essential matrix from the fundamental matrix and camera intrinsics."""
        E = self.k2.T @ self.F @ self.k1
        return E
    
    def _pose_translation_correctness(self, K, R, t):
        """Picking the correct one from the four candidates."""

        candidate_poses = [(R, t), (R, -t), (-R, t), (-R, -t)]
        # Reference point in the world coordinate
        X_world = np.random.rand(3, 1) * 10  # Random point for testing

        # The point in the camera is give by X_cam = R * X_world + t
        # We want to check if the Z coordinate is positive
        for R_cand, t_cand in candidate_poses:
            X_cam = R_cand @ X_world + t_cand
            # Projection on the camera
            X_proj = K @ X_cam
            X_proj /= X_proj[2]  # Normalize by the third homogeneous coordinate

            if X_proj.all() > 0:
                return R_cand, t_cand
        
    def _translation_from_cross_product_matrix(self, t_mat):
        """Extract translation vector from its cross-product matrix."""
        t = np.array([t_mat[2, 1], t_mat[0, 2], t_mat[1, 0]])
        return t.reshape(3, 1)
    
    def pose_estimation(self):
        """Decompose the essential matrix into rotation and translation candidates."""
        E = self._essential_matrix()
        U, _, V = np.linalg.svd(E)
        
        # Ensure a proper rotation matrix with determinant +1
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        if np.linalg.det(V) < 0:
            V[-1, :] *= -1

        # Typical essential decomposition: two possible rotations and two possible translations
        Wr = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        Wt = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

        R = U @ Wr @ V.T
        t = U @ Wt @ U.T

        t = self._translation_from_cross_product_matrix(t)

        R, t = self._pose_translation_correctness(self.k2, R, t)

        return ( np.eye(3), np.zeros((3, 1)) ), ( R, t )