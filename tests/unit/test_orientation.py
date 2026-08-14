import numpy as np

from container_id.runtime.orientation import (
    classify_orientation,
    generate_transform_candidates,
)


def test_classify_orientation():
    assert classify_orientation(None) == "near_square"
    assert classify_orientation(np.array([])) == "near_square"

    # Horizontal (w/h >= 1.5)
    assert classify_orientation(np.zeros((100, 150))) == "horizontal"
    assert classify_orientation(np.zeros((100, 200))) == "horizontal"

    # Tall (h/w >= 1.2)
    assert classify_orientation(np.zeros((120, 100))) == "tall"
    assert classify_orientation(np.zeros((200, 100))) == "tall"

    # Near square
    assert classify_orientation(np.zeros((100, 100))) == "near_square"
    assert classify_orientation(np.zeros((110, 100))) == "near_square"  # 1.1 tall
    assert classify_orientation(np.zeros((100, 140))) == "near_square"  # 1.4 wide


def test_generate_transform_candidates():
    # 2x3 horizontal crop
    horizontal_crop = np.array(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]],
        dtype=np.uint8,
    )

    cands_horiz = generate_transform_candidates(horizontal_crop, "horizontal")
    assert "identity" in cands_horiz
    assert "rotate_180" in cands_horiz
    assert "contrast_enhanced" in cands_horiz
    assert np.array_equal(cands_horiz["identity"], horizontal_crop)

    # 3x2 tall crop
    tall_crop = np.zeros((3, 2, 3), dtype=np.uint8)
    cands_tall = generate_transform_candidates(tall_crop, "tall")
    assert "identity" in cands_tall
    assert "rotate_90" in cands_tall
    assert "rotate_270" in cands_tall
    assert "vertical_unstack" not in cands_tall  # Not implemented yet

    # 2x2 square crop
    sq_crop = np.zeros((2, 2, 3), dtype=np.uint8)
    cands_sq = generate_transform_candidates(sq_crop, "near_square")
    assert "identity" in cands_sq
    assert "rotate_90" in cands_sq
    assert "rotate_180" in cands_sq
    assert "rotate_270" in cands_sq
