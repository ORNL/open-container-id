import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from container_id.runtime.crop_quality import calculate_crop_quality
from container_id.runtime.detector_onnx import ONNXDetector
from container_id.runtime.orientation import (
    classify_orientation,
    generate_transform_candidates,
)
from container_id.runtime.recognizer_onnx import ONNXRecognizer

logger = logging.getLogger(__name__)


class RuntimePipeline:
    def __init__(self, bundle_dir: str):
        self.bundle_dir = Path(bundle_dir)
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Model bundle not found: {bundle_dir}")

        # Initialize configurations (in a real system we'd load defaults from runtime_defaults.yaml)
        self.crop_padding_fraction = 0.08
        self.max_detections = 20

        logger.info(f"Loading pipeline from {bundle_dir}")
        self.detector = ONNXDetector(self.bundle_dir)
        self.recognizer = ONNXRecognizer(self.bundle_dir)

    def process_frame(self, image: np.ndarray) -> dict[str, Any]:
        """
        Process a single image frame, running detection and OCR.
        Returns a dict representing the structured result.
        """
        img_h, img_w = image.shape[:2]

        # 1. Run Detector
        detections_list = self.detector.detect([image])
        if not detections_list or not detections_list[0]:
            return {
                "model_bundle_version": "0.1.0",
                "detections": [],
                "elapsed_ms": 0.0,
            }

        detections = detections_list[0]

        # Enforce max detections config
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[
            : self.max_detections
        ]

        results = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy

            # Apply padding
            w = x2 - x1
            h = y2 - y1

            pad_x = w * self.crop_padding_fraction
            pad_y = h * self.crop_padding_fraction

            # Clamp to bounds
            c_x1 = max(0, int(x1 - pad_x))
            c_y1 = max(0, int(y1 - pad_y))
            c_x2 = min(img_w, int(x2 + pad_x))
            c_y2 = min(img_h, int(y2 + pad_y))

            crop = image[c_y1:c_y2, c_x1:c_x2]

            if crop.size == 0:
                continue

            # Crop Quality
            quality = calculate_crop_quality(crop, det.confidence)

            # Orientation & Transform
            orientation = classify_orientation(crop)
            transforms = generate_transform_candidates(crop, orientation)

            best_candidate = None

            # Run Recognizer on candidates
            for t_name, t_crop in transforms.items():
                ocr_results = self.recognizer.recognize([t_crop])
                if ocr_results and ocr_results[0]:
                    cand = ocr_results[0][0]
                    # Update transform name
                    cand.transform = t_name

                    # Very simple selection logic: prefer valid structure, then higher confidence
                    if best_candidate is None:
                        best_candidate = cand
                    else:
                        if cand.structure_valid and not best_candidate.structure_valid:
                            best_candidate = cand
                        elif (
                            cand.structure_valid == best_candidate.structure_valid
                            and cand.confidence > best_candidate.confidence
                        ):
                            # both valid or both invalid, prefer confidence
                            best_candidate = cand

            det_res = {
                "bbox_xyxy": [float(c_x1), float(c_y1), float(c_x2), float(c_y2)],
                "detector_confidence": float(det.confidence),
                "crop_quality": float(quality),
                "orientation": orientation,
            }

            if best_candidate:
                det_res["best_candidate"] = best_candidate.model_dump()

            results.append(det_res)

        return {
            "model_bundle_version": "0.1.0",
            "detections": results,
            "elapsed_ms": 0.0,  # Time tracking not implemented
        }

    def infer_image(self, image_path: str, output_path: str | None = None) -> dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = self.process_frame(img)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        return result

    def infer_directory(
        self, input_dir: str, output_path: str, recursive: bool = False
    ) -> None:
        p = Path(input_dir)
        if not p.exists() or not p.is_dir():
            raise NotADirectoryError(f"Directory not found: {input_dir}")

        files = []
        extensions = [".jpg", ".jpeg", ".png", ".webp"]

        if recursive:
            for ext in extensions:
                files.extend(
                    list(p.rglob(f"*{ext}")) + list(p.rglob(f"*{ext.upper()}"))
                )
        else:
            for ext in extensions:
                files.extend(list(p.glob(f"*{ext}")) + list(p.glob(f"*{ext.upper()}")))

        # Deterministic sorting required
        files = sorted(files)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for fpath in files:
                try:
                    res = self.infer_image(str(fpath))
                    res["file"] = str(fpath.relative_to(p))
                    f.write(json.dumps(res) + "\n")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error processing {fpath}: {e}")
                    f.write(
                        json.dumps({"file": str(fpath.relative_to(p)), "error": str(e)})
                        + "\n"
                    )
