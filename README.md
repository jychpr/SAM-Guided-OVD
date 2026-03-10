# SAM-Guided-OVD: Open-Vocabulary DETR with Class-Agnostic Proposals for Grounded Robotic VQA

This repository contains the official implementation of the Master's Thesis: **Enhancing Open-Vocabulary DETR with Class-Agnostic SAM Proposals for Grounded Robotic VQA.**

## Abstract
Standard Open-Vocabulary Object Detectors (OV-OD) often struggle with fine-grained domain shifts, such as identifying highly specific, novel industrial tools in robotic assembly environments. This project introduces a unified architecture that leverages the geometric precision of Fast Segment Anything (FastSAM) as a dynamic Region Proposal Network (RPN) for a Denoising DETR backbone (OV-DQUO). By injecting deterministic bounding box priors into the DETR decoder, the model guarantees the localization of novel objects, subsequently generating highly accurate semantic features used for Grounded Visual Question Answering (VQA).

## Architecture
1. **Geometric Prior Generation:** FastSAM generates dense, class-agnostic bounding box proposals.
2. **Dynamic Query Formulation:** Proposals are injected as `reference_points` into the OV-DQUO decoder.
3. **Open-Vocabulary Classification:** Denoising text queries strictly classify the isolated objects.
4. **Grounded VQA:** Stabilized object queries cross-attend with text tokens to answer operator prompts.

## Modern Setup Guide (RTX 5090 / CUDA 12.8 / Python 3.10)

**Warning for Flagship GPU Users:** The original OV-DQUO repository relies on obsolete PyTorch 2.0 binaries that do not support Ada/Blackwell architectures (`sm_120`). Stable PyTorch releases currently lack the memory allocation binaries for these cards. This guide establishes a bleeding-edge environment using Native CUDA 12 compilation.

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
pip install lvis pycocotools scipy shapely pandas opencv-python tqdm timm submitit einops transformers open_clip_torch torchmetrics mmcv==1.7.1 termcolor yapf==0.32.0
```

**4. Inject OV-DQUO Source Code Safely**
We clone the original architecture but only move the internal modules to prevent overwriting custom repository files (like this README or custom inference scripts).
```bash
git clone https://github.com/xiaomoguhz/ov-dquo.git temp_dquo

# Move only the core architecture modules
mv temp_dquo/models ./ 
mv temp_dquo/datasets ./ 
mv temp_dquo/util ./ 
mv temp_dquo/engine.py ./ 
mv temp_dquo/main.py ./ 
mv temp_dquo/config ./ 
mv temp_dquo/custom_tools ./

# Clean up
rm -rf temp_dquo
```

**5. Patch the Obsolete C++ Code**
The original authors used a deprecated PyTorch C++ API (value.type().is_cuda()) that has been entirely removed from modern PyTorch. You must patch their source code before compiling.
```bash
# 1. Fix CUDA assertions
find models/ops/src -type f -exec sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' {} +

# 2. Fix legacy type dispatch macros
find models/ops/src -type f -exec sed -i 's/value\.type()/value.scalar_type()/g' {} +
```

**6. Native Architecture Compilation**
Force the NVCC compiler to target your specific GPU architecture (sm_120), and compile Detectron2 without allowing it to overwrite your Nightly PyTorch engine.
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
- Pretrained weights: pretrained/region_prompt_R50x4.pth
- Weights: ckpt/OVDQUO_RN50x4_COCO.pth
- Images: data/gai19coco/test/
- JSON: data/gai19coco/test/_annotations.coco.json

**2. Custom Dataset Inference Caveats**
If you run inference on a custom dataset (not COCO/LVIS), you must apply two overrides in your evaluation script (eval_ovdquo.py):
- CPU Normalization: Normalize images on the CPU (TF.normalize) before pushing to the GPU to avoid missing torchvision CUDA kernels.
- Bypass Benchmark Scaling: Set args.target_class_factor = 1.0 after loading the config to prevent the transformer from crashing when evaluating non-standard class counts.

**3. Run the Baseline.**
```bash
python eval_ovdquo.py
```
