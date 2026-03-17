# SAM-Guided-OVD: Open-Vocabulary DETR with Hybrid SAM Proposals for Grounded Robotic VQA

This repository contains the official implementation of the Master's Thesis: **Enhancing Open-Vocabulary DETR with Class-Agnostic SAM Proposals for Grounded Robotic VQA.**

## Abstract
Standard Open-Vocabulary Object Detectors (OV-OD) often struggle with fine-grained domain shifts, such as identifying highly specific, novel industrial tools in robotic assembly environments. This project introduces a unified architecture that leverages the geometric precision of Fast Segment Anything (FastSAM) as a dynamic Region Proposal Network (RPN) for a Denoising DETR backbone (OV-DQUO). 

By utilizing a **V5 Hybrid RPN** with Nearest-Neighbor feature matching and **Textual Concept Expansion**, this architecture achieves a **23.11% mAP@0.50** in zero-shot localization of novel industrial tools, outperforming standard baseline architectures (GDINO, COOP, and raw OV-DQUO) entirely without fine-tuning.

## Architectural Upgrades (How this differs from baseline OV-DQUO)
This repository is a heavily modified fork of the original OV-DQUO. The following permanent upgrades have been integrated directly into the source code:
1. **Geometric Prior Generation:** FastSAM generates dense, class-agnostic bounding box proposals.
2. **V5 Hybrid RPN (`models/transformer/ov_deformable_transformer.py`):** SAM proposals are geometrically matched to DETR's multi-scale encoder features. Remaining query slots are populated with native high-confidence semantic guesses to prevent attention collapse.
3. **Textual Concept Expansion (`eval_thesis.py`):** Aggressive linguistic surface area descriptions are passed to the frozen CLIP text encoder to maximize visual-semantic alignment.
4. **Legacy C++ CUDA Fixes (`models/ops/src/`):** Obsolete PyTorch C++ APIs (`value.type().is_cuda()`) have been permanently patched to support modern CUDA 12.x compilation.

---

## Modern Setup Guide (RTX 5090 / CUDA 12.8 / Python 3.10)

**Warning for Flagship GPU Users:** Stable PyTorch releases currently lack the memory allocation binaries for Ada/Blackwell architectures (`sm_120`). This guide establishes a bleeding-edge environment using Native CUDA 12 compilation.

**1. Create the Clean Environment**
Do not use Python 3.9. Use 3.10 to ensure compatibility with modern typing and newer wheels.
```bash
conda create -n sgo python=3.10 -y
conda activate sgo
```

**2. Install the Bleeding-Edge Engine**
You must use the Nightly cu128 (or cu126) wheels to get native RTX 5090 support.
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**3. Install Meta-AI Bridges & Project Dependencies**
Bypass pip's dependency resolver by installing the strict foundational requirements for Detectron2 before compiling the vision stack.
```bash
# Meta foundational libraries
pip install fvcore iopath omegaconf cloudpickle black hydra-core tensorboard

# Standard vision/evaluation stack
pip install lvis pycocotools scipy shapely pandas opencv-python tqdm timm submitit einops transformers open_clip_torch torchmetrics mmcv==1.7.1 termcolor yapf==0.32.0 ultralytics segment-anything
```

**4. Native Architecture Compilation**
Because the legacy C++ code is already patched in this repository, you simply need to force the NVCC compiler to target your specific GPU architecture (sm_120) and compile the Deformable Attention operations natively.
```bash
# Target RTX 5090 natively
export TORCH_CUDA_ARCH_LIST="12.0"

# Build Custom Deformable Attention
cd models/ops
rm -rf build/ dist/
sh make.sh
cd ../../

# Build Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation --force-reinstall --no-deps
```

## Dataset Preparation & Inference

**1. Data Staging**
Ensure your files are placed exactly here:
- FastSAM Weights: FastSAM-x.pt (Root directory, auto-downloads if missing)
- Pretrained weights: pretrained/region_prompt_R50x4.pth
- Weights: ckpt/OVDQUO_RN50x4_COCO.pth
- Images: data/gai19coco/test/
- JSON: data/gai19coco/test/_annotations.coco.json

**2. Run the Thesis Evaluation**
The evaluation script dynamically loads FastSAM, executes the V5 Hybrid RPN injection, utilizes the expanded textual prompt map, and computes the metrics.
```bash
python eval_thesis.py
```
