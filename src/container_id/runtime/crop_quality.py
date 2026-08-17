import cv2
import numpy as np


def calculate_crop_quality(crop: np.ndarray, detector_confidence: float = 1.0) -> float:
    """
    Evaluates a crop image based on pixel dimensions, Laplacian blur score,
    contrast, aspect ratio, and detector confidence to yield a combined quality score (0.0 to 1.0).
    """
    if crop is None or crop.size == 0:
        return 0.0

    h, w = crop.shape[:2]

    # 1. Pixel dimensions (penalize if too small, e.g. w < 48 or h < 16)
    # Using a soft sigmoid-like penalty or linear ramp.
    min_w, min_h = 48, 16
    dim_score = min(1.0, (w / min_w) * 0.5 + (h / min_h) * 0.5)

    # Convert to grayscale for variance/contrast metrics
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # 2. Blur (Laplacian variance)
    # A standard sharp image often has variance > 100. Let's cap at 500 for normalization.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, laplacian_var / 500.0)

    # 3. Contrast (Standard deviation of intensity)
    # High contrast text often has std > 40
    std_dev = np.std(gray)
    contrast_score = min(1.0, std_dev / 60.0)

    # 4. Aspect Ratio (penalize extreme perspective / clipping)
    # Assuming text should somewhat fit typical bounding box aspect ratios
    aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 0
    # Extreme aspect ratio > 15 might be a sliver of the container rather than the text.
    aspect_score = (
        1.0 if aspect_ratio < 10 else max(0.0, 1.0 - (aspect_ratio - 10) / 10.0)
    )

    # Combine scores. We want to weight detector confidence high.
    quality = (
        0.3 * detector_confidence
        + 0.2 * dim_score
        + 0.2 * blur_score
        + 0.2 * contrast_score
        + 0.1 * aspect_score
    )

    return float(np.clip(quality, 0.0, 1.0))
