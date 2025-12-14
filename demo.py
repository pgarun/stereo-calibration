"""Example: end-to-end stereo pipeline using ORB fallback (no torch required).

This script will detect ORB features, match them, estimate the fundamental
matrix using `estimator.estimate_fundamental_matrix`, compute the essential
matrix via `calibration.StereoCalibration`, and print the two pose candidates.

Usage:
    python examples/run_example.py --left left.jpg --right right.jpg
"""

import argparse
import cv2
import numpy as np
from estimator import estimate_fundamental_matrix
from calibration import StereoCalibration


def orb_match_pairs(imgL, imgR, max_kp=2000, topk=500):
    """Detect ORB features and return matched keypoint pairs.

    Returns list of ((x1,y1),(x2,y2))
    """
    orb = cv2.ORB_create(nfeatures=max_kp)
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    if des1 is None or des2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    pairs = []
    for m in matches[:topk]:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        pairs.append((pt1, pt2))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', '-l', required=True)
    parser.add_argument('--right', '-r', required=True)
    parser.add_argument('--fx', type=float, default=700.0)
    parser.add_argument('--fy', type=float, default=None)
    parser.add_argument('--cx', type=float, default=320.0)
    parser.add_argument('--cy', type=float, default=240.0)
    parser.add_argument('--topk', type=int, default=500)
    args = parser.parse_args(argv)

    imgL = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise SystemExit('Could not read input images')

    pairs = orb_match_pairs(imgL, imgR, topk=args.topk)
    if len(pairs) < 8:
        raise SystemExit('Not enough matches for fundamental estimation')

    F = estimate_fundamental_matrix(pairs)

    fx = args.fx
    fy = args.fy or fx
    k = np.array([[fx, 0, args.cx], [0, fy, args.cy], [0, 0, 1]])

    stereo = StereoCalibration(k, k, F)
    poses = stereo.pose_estimation()

    (R1, t1), (R2, t2) = poses
    print('Pose 1:')
    print(R1)
    print(t1)
    print('\nPose 2:')
    print(R2)
    print(t2)


if __name__ == '__main__':
    main()