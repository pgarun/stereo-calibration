import cv2
from feature import feature
import numpy as np

try:
    import torch
    import kornia.feature as KF
except ImportError:
    torch = None
    KF = None

from feature import feature
from calibration import StereoCalibration
from estimator import estimate_fundamental_matrix, residuals

def stereo_calibration(img1, img2, k1, k2):
    """Perform stereo calibration between two images."""
    # Initialize feature extractor and matcher
    feat_extractor = feature(device='cuda' if torch and torch.cuda.is_available() else 'cpu')

    # Extract features
    feats1 = feat_extractor.extract_features(img1)
    feats2 = feat_extractor.extract_features(img2)

    # Match features
    matches = feat_extractor.match_features(feats1, feats2)

    # Estimate fundamental matrix
    F = estimate_fundamental_matrix(matches)

    # Perform stereo calibration
    stereo_calib = StereoCalibration(k1, k2, F)
    (R1, t1), (R2, t2) = stereo_calib.decompose_essential_matrix()

    return (R1, t1), (R2, t2)

if __name__ == "__main__":
    # NOTE: Replace with actual image paths and intrinsic matrices
    img1 = cv2.imread('')
    img2 = cv2.imread('right_image.jpg')

    # Intrinsic matrices
    k1 = np.array([[700, 0, 320],
                   [0, 700, 240],
                   [0, 0, 1]])
    k2 = np.array([[700, 0, 320],
                   [0, 700, 240],
                   [0, 0, 1]])

    (R1, t1), (R2, t2) = stereo_calibration(img1, img2, k1, k2)

    print("Pose 1: R =", R1, ", t =", t1)
    print("Pose 2: R =", R2, ", t =", t2)