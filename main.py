import argparse
import cv2
import numpy as np
import sys

try:
    import torch
    import kornia.feature as KF
except Exception:
    torch = None
    KF = None

from feature import feature as Feature
from calibration import StereoCalibration
from estimator import estimate_fundamental_matrix, residuals


def stereo_calibration(img1, img2, k1, k2, device=None, topk=500):
    """Perform stereo calibration between two images.

    Args:
        img1, img2: image arrays (ndarray) or paths (str)
        k1, k2: 3x3 intrinsic matrices (ndarray)
        device: 'cpu' or 'cuda' or None (auto)
        topk: number of top matches to use for fundamental estimation
    Returns:
        Two pose tuples: (R1, t1), (R2, t2)
    """
    if device is None:
        device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'

    feat_extractor = Feature(device=device)

    # feature_extraction accepts either image path or ndarray
    feats1 = feat_extractor.feature_extraction(img1)
    feats2 = feat_extractor.feature_extraction(img2)

    matches = feat_extractor.feature_matching(feats1, feats2)
    if len(matches) == 0:
        raise RuntimeError("No matches found between images")

    # keep top-k matches for estimation
    top_pairs = feat_extractor.topk_matches(matches, min(topk, len(matches)))

    # estimate fundamental matrix
    F = estimate_fundamental_matrix(top_pairs)

    stereo_calib = StereoCalibration(k1, k2, F)
    (R1, t1), (R2, t2) = stereo_calib.pose_estimation()

    return (R1, t1), (R2, t2)


def parse_intrinsics(args):
    # Build a simple intrinsic matrix from focal/cx/cy or use defaults
    fx = args.fx or 700.0
    fy = args.fy or fx
    cx = args.cx or 320.0
    cy = args.cy or 240.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stereo calibration pipeline (feature -> F -> essential -> poses)')
    parser.add_argument('--left', '-l', required=True, help='Left image path')
    parser.add_argument('--right', '-r', required=True, help='Right image path')
    parser.add_argument('--fx', type=float, default=700.0, help='Focal length x (default 700)')
    parser.add_argument('--fy', type=float, default=None, help='Focal length y (default = fx)')
    parser.add_argument('--cx', type=float, default=320.0, help='Principal point x (default 320)')
    parser.add_argument('--cy', type=float, default=240.0, help='Principal point y (default 240)')
    parser.add_argument('--device', type=str, default=None, help='Device for SuperPoint/SuperGlue (cpu/cuda)')
    parser.add_argument('--topk', type=int, default=500, help='Top-k matches to use for F estimation')

    args = parser.parse_args(argv)

    imgL = cv2.imread(args.left, cv2.IMREAD_UNCHANGED)
    imgR = cv2.imread(args.right, cv2.IMREAD_UNCHANGED)

    if imgL is None:
        print(f"Failed to read left image: {args.left}")
        sys.exit(2)
    if imgR is None:
        print(f"Failed to read right image: {args.right}")
        sys.exit(2)

    k1 = parse_intrinsics(args)
    k2 = parse_intrinsics(args)

    poses = stereo_calibration(imgL, imgR, k1, k2, device=args.device, topk=args.topk)

    (R1, t1), (R2, t2) = poses
    print("Pose 1: R=\n", R1)
    print("t=", t1)
    print("---")
    print("Pose 2: R=\n", R2)
    print("t=", t2)


if __name__ == '__main__':
    main()