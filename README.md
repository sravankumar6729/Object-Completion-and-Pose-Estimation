# Object Completion and Pose Estimation

### OcclusionPoseNet v5 — A Smart Pipeline for Amodal Completion and Orientation Estimation from Partial Visual Observations

This repository contains the implementation of **OcclusionPoseNet v5**, a two-stage deep learning pipeline that reconstructs occluded objects and estimates their orientation from partial visual observations. The work addresses a core challenge in robotics perception and automated spatial reasoning: recovering reliable pose estimates when objects are partially hidden by clutter or other workpieces.

---

## Overview

Traditional 6D pose estimation methods rely on fully visible object geometry and degrade sharply under heavy occlusion. OcclusionPoseNet v5 tackles this by combining:

1. **Amodal completion** — reconstructing the full geometry of a partially occluded object
2. **Orientation classification** — predicting object pose using global attention over the completed geometry
3. **Smart routing** — skipping the reconstruction stage entirely for non-occluded inputs, saving compute

The result is a pipeline that is both more accurate under occlusion and more efficient on clean inputs — making it suitable for edge devices with restricted compute budgets.

---

## Architecture

The pipeline consists of two stages:

### Stage 1 — Amodal Completion (GELU-UNet)
- U-Net encoder-decoder with skip connections
- 4 downsampling blocks with 3×3 convolutions
- **GELU activations** instead of ReLU, for smoother gradient flow and better recovery of edge details in occluded regions
- Decoder outputs reconstructed geometry at 128×128 resolution
- Optimized with a hybrid loss combining pixel-level accuracy and structural similarity:

```
L_total = α · L_MSE + (1 − α) · L_SSIM
```

### Stage 2 — Orientation Estimation (Deepened ViT)
- Vision Transformer with 16×16 patch embeddings and multi-head self-attention
- Classifies orientation into four categories: **0°, 90°, 180°, 270°**
- **3-layer deepened MLP head** for improved discrimination between visually similar rotations (e.g., 90° vs. 270° under symmetry)

### Smart Routing (Logic Gate)
A lightweight occlusion detector computes the ratio of occluded pixels in the input. Images below the occlusion threshold τ bypass Stage 1 entirely and go straight to the ViT for pose estimation, avoiding unnecessary reconstruction compute.

```
D(I) = Stage 1 → Stage 2   if occlusion ratio > τ
     = Stage 2 only        otherwise
```

---

## Results

| Metric                      | Baseline (v2) | Proposed (v5) |
|------------------------------|:---:|:---:|
| Input resolution              | 96 × 96 | 128 × 128 |
| Training time (hours)         | ~5.0 | **1.5** |
| Completeness quality (SSIM)   | 0.76 | **0.92** |
| Pose estimation accuracy (%)  | 74.2% | **89.5%** |

**Ablation — Amodal Completion Variants**

| Model variant        | PSNR (dB) | SSIM | MSE |
|-----------------------|:---:|:---:|:---:|
| v2 baseline            | 22.4 | 0.76 | 0.045 |
| v5 with ReLU-U-Net      | 26.8 | 0.88 | 0.012 |
| **v5 with GELU-U-Net**  | **29.5** | **0.92** | **0.008** |

**Inference Speed (Fast Path)**

| Processing Step             | Time (ms) | GPU Memory (GB) |
|------------------------------|:---:|:---:|
| Object Detection Gate         | 4.2 | 0.5 |
| Image Completion (GELU-UNet)  | 18.5 | 2.1 |
| 3D Pose Estimation (ViT)      | 12.8 | 1.4 |
| **Total**                     | **35.5** | **4.0** |

Trained and evaluated on **30,000 images** (CIFAR-10, CIFAR-100, and STL-10 combined) with synthetic occlusion masks covering 8–60% of the image area.

---

## Tech Stack

- **Framework:** PyTorch
- **Model components:** `timm` (Vision Transformer backbone), custom U-Net
- **Training:** Automatic Mixed Precision (AMP), Adam optimizer, gradient clipping (max norm 1.0)
- **Demo/UI:** Gradio
- **Supporting libraries:** OpenCV, scikit-image, Pillow, NumPy, Matplotlib, einops
- **Hardware:** NVIDIA T4 GPU (16GB VRAM), Google Colab

---

## Repository Contents

```
├── occlusionposenet_v5_final.py   # Full pipeline (Colab-exported script)
├── research_paper.pdf             # Accompanying research paper
└── README.md
```

The core script includes:
- `models/unet.py` — GELU-UNet for amodal completion
- `models/vit_pose.py` — ViT-based pose classifier with deepened MLP head
- `utils/dataset_loader.py` — Combined CIFAR-10/100 + STL-10 loader with synthetic occlusion generation
- Training loops for both stages (AMP, gradient clipping)
- `demo/app.py` — Interactive Gradio demo with adjustable synthetic occlusion

---

## Running the Project

This project was developed and run in **Google Colab** (T4 GPU runtime).

1. Open the notebook in Colab and set **Runtime → Change runtime type → T4 GPU**
2. Run the setup cells to install dependencies and write model/utility files
3. Run the training cells for the U-Net and ViT stages
4. Launch the Gradio demo cell to interactively test the pipeline on uploaded images, with an option to force synthetic occlusion at a chosen percentage

---

## Limitations & Future Work

- The pipeline currently does not perform well on **multi-body occlusion**, where multiple target objects overlap and occlude each other
- Pose estimation is currently limited to 2D orientation (SO(2)); planned extension to full **6D pose estimation** (SO(3) + translation) using RGB-D depth input
- Future work includes **Graph Neural Networks (GNNs)** to model spatial relationships between multiple occluding objects, and **temporal cross-attention** to handle moving occluders in video streams

---

## Citation

If you use this work, please cite:

```
Mallela, S., Nallagutla, H., K.S.S.V., S. K., & Ponnam, S. (2025).
Object Completion and Pose Estimation from Partial Visual Observations using Deep Learning.
VIT-AP University.
```

---

## Authors

- Dr. Mallela Sivanagaraju — VIT-AP University
- Nallagutla Harivyshnavi — VIT-AP University
- K.S.S.V. Sravan Kumar — VIT-AP University
- Ponnam Shivamani — VIT-AP University
