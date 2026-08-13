import numpy as np
from container_id.runtime.crop_quality import calculate_crop_quality


def test_calculate_crop_quality_empty():
    assert calculate_crop_quality(None) == 0.0
    assert calculate_crop_quality(np.array([])) == 0.0


def test_calculate_crop_quality_ideal():
    # Make a synthetic "perfect" crop: high contrast, sharp edge
    # Let's make it 64x256
    crop = np.zeros((64, 256, 3), dtype=np.uint8)
    crop[:, 100:150] = 255  # sharp white block on black background

    # Needs some noise so standard deviation / laplacian triggers
    noise = np.random.randint(0, 50, (64, 256, 3), dtype=np.uint8)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    score = calculate_crop_quality(crop, detector_confidence=1.0)
    assert 0.5 < score <= 1.0  # Should be reasonably high


def test_calculate_crop_quality_blurry_low_contrast():
    # solid color = 0 contrast, 0 blur
    crop = np.ones((64, 256, 3), dtype=np.uint8) * 128
    score = calculate_crop_quality(crop, detector_confidence=0.5)

    # detector (0.3*0.5) + dim (0.2*1.0) + blur(0.0) + contrast(0.0) + aspect (0.1*1.0) = 0.15 + 0.2 + 0.1 = 0.45
    assert np.isclose(score, 0.45, atol=0.01)


def test_calculate_crop_quality_tiny():
    # Tiny crop
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    score = calculate_crop_quality(crop, detector_confidence=0.9)
    # Dimension score should be low
    assert score < 0.6
