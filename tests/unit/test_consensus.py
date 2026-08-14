from datetime import datetime, timezone, timedelta
from container_id.runtime.tracking import Track
from container_id.runtime.consensus import ConsensusEngine


def test_consensus_engine_emit():
    engine = ConsensusEngine(
        minimum_supporting_frames=3,
        minimum_weighted_score=2.0,
        duplicate_suppression_seconds=30.0,
    )

    t0 = datetime.now(timezone.utc)
    track = Track((0, 0, 10, 10), t0)

    # 1. Provide insufficient frames
    track.detector_confidences = [0.9, 0.9]
    track.crop_quality_scores = [0.9, 0.9]
    track.ocr_candidates = [
        {
            "timestamp": t0,
            "candidate": {
                "normalized_text": "BMOU1234567",
                "confidence": 0.9,
                "structure_valid": True,
                "check_digit_valid": True,
            },
        },
        {
            "timestamp": t0 + timedelta(seconds=1),
            "candidate": {
                "normalized_text": "BMOU1234567",
                "confidence": 0.9,
                "structure_valid": True,
                "check_digit_valid": True,
            },
        },
    ]

    event = engine.process_track(track, t0 + timedelta(seconds=1))
    assert event is None

    # 2. Add 3rd frame, should trigger
    track.detector_confidences.append(0.9)
    track.crop_quality_scores.append(0.9)
    track.ocr_candidates.append(
        {
            "timestamp": t0 + timedelta(seconds=2),
            "candidate": {
                "normalized_text": "BMOU1234567",
                "confidence": 0.9,
                "structure_valid": True,
                "check_digit_valid": True,
            },
        }
    )

    event = engine.process_track(track, t0 + timedelta(seconds=2))
    assert event is not None
    assert event.container_number == "BMOU1234567"
    assert event.supporting_frames == 3
    assert track.emitted_event is True


def test_duplicate_suppression():
    engine = ConsensusEngine(
        minimum_supporting_frames=1,  # reduce for quick test
        minimum_weighted_score=0.1,
        duplicate_suppression_seconds=30.0,
        camera_id="cam1",
    )

    t0 = datetime.now(timezone.utc)
    track1 = Track((0, 0, 10, 10), t0)
    track1.detector_confidences = [0.9]
    track1.crop_quality_scores = [0.9]
    track1.ocr_candidates = [
        {
            "timestamp": t0,
            "candidate": {
                "normalized_text": "TEST1234567",
                "confidence": 0.9,
                "structure_valid": True,
                "check_digit_valid": True,
            },
        }
    ]

    # First track emits
    event1 = engine.process_track(track1, t0)
    assert event1 is not None

    # Second track with same text, same camera, 5 seconds later
    t1 = t0 + timedelta(seconds=5)
    track2 = Track((0, 0, 10, 10), t1)
    track2.detector_confidences = [0.9]
    track2.crop_quality_scores = [0.9]
    track2.ocr_candidates = [
        {
            "timestamp": t1,
            "candidate": {
                "normalized_text": "TEST1234567",
                "confidence": 0.9,
                "structure_valid": True,
                "check_digit_valid": True,
            },
        }
    ]

    # Suppressed
    event2 = engine.process_track(track2, t1)
    assert event2 is None

    # Third track 35 seconds later (past suppression window)
    t2 = t0 + timedelta(seconds=35)
    track3 = Track((0, 0, 10, 10), t2)
    track3.detector_confidences = [0.9]
    track3.crop_quality_scores = [0.9]
    track3.ocr_candidates = [
        {
            "timestamp": t2,
            "candidate": {
                "normalized_text": "TEST1234567",
                "confidence": 0.9,
                "structure_valid": True,
                "check_digit_valid": True,
            },
        }
    ]

    event3 = engine.process_track(track3, t2)
    assert event3 is not None  # Allowed through
