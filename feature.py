import cv2
import numpy as np

try:
    import torch
    import kornia.feature as KF
except ImportError:
    torch = None
    KF = None

class feature:
    """SuperPoint extractor and SuperGlue matcher wrapper."""
    def __init__(self, device=None, max_keypoints=2048, detection_threshold=0.005, superglue_weights='outdoor'):
        self.device = device
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold
        self.superglue_weights = superglue_weights

        # Lazy-loaded neural models
        self._superpoint = None
        self._superglue = None

    def _ensure_super_models(self):
        """Lazily create SuperPoint/SuperGlue modules on the chosen device."""

        if KF is None or torch is None:
            raise ImportError("SuperPoint/SuperGlue requires torch and kornia.feature. Install with: pip install torch kornia")

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self._superpoint is None:
            self._superpoint = KF.SuperPoint(max_num_keypoints=self.max_keypoints, detection_threshold=self.detection_threshold).to(self.device).eval()

        if self._superglue is None:
            self._superglue = KF.SuperGlue(KF.superglue_settings[self.superglue_weights]).to(self.device).eval()

    def _to_gray_tensor(self, img):
        """Convert ndarray image to grayscale float tensor on the target device."""

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        tensor = torch.from_numpy(gray).float() / 255.0
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def _superpoint_extract(self, img):
        """Run SuperPoint and return serialized keypoints, descriptors, scores, and tensors."""

        self._ensure_super_models()
        tensor = self._to_gray_tensor(img)
        with torch.inference_mode():
            feats = self._superpoint({"image": tensor})

        kp_tensor = feats["keypoints"][0]  # (N, 2)
        desc_tensor = feats["descriptors"][0]  # (256, N)
        score_tensor = feats["scores"][0]  # (N,)

        kp_np = kp_tensor.cpu().numpy()
        desc_np = desc_tensor.permute(1, 0).cpu().numpy()
        scores_np = score_tensor.cpu().numpy()

        return {
            "serialized_keypoints": [(float(x), float(y)) for x, y in kp_np],
            "descriptors": desc_np,
            "scores": scores_np,
            "torch": {
                "keypoints": kp_tensor.unsqueeze(0),
                "descriptors": desc_tensor.unsqueeze(0),
                "scores": score_tensor.unsqueeze(0),
            },
        }

    def feature_extraction(self, image_path):
        """
        Extract features using SuperPoint.

        Args:
            image_path: path to an image file (str) or an ndarray (BGR or grayscale).

        Returns:
            keypoints: list of tuples (x, y)
            descriptors: ndarray of shape (N, 256)
            aux: dict containing torch tensors for SuperGlue
        """
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read image from path: {image_path}")
        else:
            img = image_path

        if img is None:
            raise ValueError("Input image is None")

        feats = self._superpoint_extract(img)
        return feats

    def feature_matching(self, feats1, feats2):
        """
        Match SuperPoint features using SuperGlue.

        Args:
            feats1: aux dict returned by feature_extraction for image 1
            feats2: aux dict returned by feature_extraction for image 2

        Returns:
            list of tuples: (kp0_xy, kp1_xy, score)
        """
        if not isinstance(feats1, dict) or not isinstance(feats2, dict):
            raise ValueError("Pass the aux dict returned by feature_extraction for each image.")

        self._ensure_super_models()
        with torch.inference_mode():
            out = self._superglue({
                "keypoints0": feats1["torch"]["keypoints"],
                "keypoints1": feats2["torch"]["keypoints"],
                "descriptors0": feats1["torch"]["descriptors"],
                "descriptors1": feats2["torch"]["descriptors"],
                "scores0": feats1["torch"]["scores"],
                "scores1": feats2["torch"]["scores"],
            })

        matches0 = out["matches0"][0].cpu().numpy()
        scores0 = out["matching_scores0"][0].cpu().numpy()

        matches = []
        kp0 = feats1["serialized_keypoints"]
        kp1 = feats2["serialized_keypoints"]
        for idx0, idx1 in enumerate(matches0):
            if idx1 == -1:
                continue
            matches.append((kp0[idx0], kp1[idx1], float(scores0[idx0])))
        return matches
    
    def topk_matches(self, matches, k):
        """
        Get top-k matches based on scores.

        Args:
            matches: list of tuples (kp0_xy, kp1_xy, score)
            k: number of top matches to return

        Returns:
            list of top-k matches
        """
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
        return [(m[0], m[1]) for m in sorted_matches[:k]]