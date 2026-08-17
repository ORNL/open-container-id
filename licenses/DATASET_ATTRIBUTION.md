# Dataset Attribution

The models in this repository are trained using publicly available datasets. We are deeply grateful to the original creators for open-sourcing their data.

## 1. PranW Container Number Detection
- **Creator:** PranW
- **Dataset:** Container Number Detection, Version 7
- **Source URL:** [https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/7](https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/7)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Access Date:** 2024 (Exact date varies by model release)
- **Modifications:** The dataset was audited to enforce strict class semantics (ensuring `objects` maps to `container_number`). Duplicate images were removed or clustered to prevent train/test leakage. The dataset was completely re-split, ignoring the original split, to ensure a clean evaluation set.

## 2. dasad Container number
- **Creator:** dasad
- **Dataset:** Container number, Version 1
- **Source URL:** [https://universe.roboflow.com/dasad/container-number-pmov4-tvflz/dataset/1](https://universe.roboflow.com/dasad/container-number-pmov4-tvflz/dataset/1)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Access Date:** 2024 (Exact date varies by model release)
- **Modifications:** The dataset was used primarily to extract high-resolution text crops for OCR training. Cropping and augmentations were applied to text regions. Duplicate images were removed or clustered to prevent train/test leakage. The dataset was completely re-split.

## License Implication for Trained Weights
Because the source datasets are licensed under CC BY 4.0, any weights trained on them inherit the requirement for attribution. If you distribute models trained using this repository's default configurations, you **must** include this attribution notice in your distribution.
