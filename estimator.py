import numpy as np

def estimate_fundamental_matrix(matches):
    """
    Estimate the fundamental matrix from matched keypoints
    using 8-point algorithm.

    Args:
        matches: list of tuples (kp0_xy, kp1_xy)
    """
    if len(matches) < 8:
        raise ValueError("Number of matches are lesser than 8, cannot compute fundamental matrix.")

    # Extract keypoints from matches
    kp0 = np.array([m[0] for m in matches])
    kp1 = np.array([m[1] for m in matches])

    # Setup Homogenous linear equation
    # x2' * F * x1 = 0
    A = []
    for i in range(len(kp0)):
        x1, y1 = kp0[i]
        x2, y2 = kp1[i]
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    A = np.array(A)

    # Solve for F
    Q, _ = np.linalg.qr(A.T)
    F = Q[:, -1].reshape((3, 3), order='C')

    # Enforce rank 2 constraint
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V.T

    return F

def residuals(F, matches):
    """
    Compute residuals for the fundamental matrix estimation.

    Args:
        F: Fundamental matrix
        matches: list of tuples (kp0_xy, kp1_xy)
    """
    res = 0
    for m in matches:
        x1 = np.array([m[0][0], m[0][1], 1])
        x2 = np.array([m[1][0], m[1][1], 1])
        res += abs(x2.T @ F @ x1)
    return res