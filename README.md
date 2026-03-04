# SAM-Guided-OVD: Open-Vocabulary DETR with Class-Agnostic Proposals for Grounded Robotic VQA

This repository contains the official implementation of the Master's Thesis: **Enhancing Open-Vocabulary DETR with Class-Agnostic SAM Proposals for Grounded Robotic VQA.**

## Abstract
Standard Open-Vocabulary Object Detectors (OV-OD) often struggle with fine-grained domain shifts, such as identifying highly specific, novel industrial tools in robotic assembly environments. This project introduces a unified architecture that leverages the geometric precision of Fast Segment Anything (FastSAM) as a dynamic Region Proposal Network (RPN) for a Denoising DETR backbone (OV-DQUO). By injecting deterministic bounding box priors into the DETR decoder, the model guarantees the localization of novel objects, subsequently generating highly accurate semantic features used for Grounded Visual Question Answering (VQA).

## Architecture
1. **Geometric Prior Generation:** FastSAM generates dense, class-agnostic bounding box proposals.
2. **Dynamic Query Formulation:** Proposals are injected as `reference_points` into the OV-DQUO decoder.
3. **Open-Vocabulary Classification:** Denoising text queries strictly classify the isolated objects.
4. **Grounded VQA:** Stabilized object queries cross-attend with text tokens to answer operator prompts.

## Setup & Installation

**Warning for Flagship GPU Users (RTX 5090 / Blackwell):** Standard PyTorch 2.x wheels do not currently contain compiled binaries for the `sm_120` architecture. You must use the Nightly build and compile C++ extensions from source with the PTX flag enabled.

**1. Create the Environment**
```bash
conda create -n sgo python=3.10 -y
conda activate sgo
```

**2. Install Core Deep Learning Engine (Nightly + Vision Source)**
```bash
pip install --pre torch --index-url [https://download.pytorch.org/whl/nightly/cu124](https://download.pytorch.org/whl/nightly/cu124)
export TORCH_CUDA_ARCH_LIST="9.0;9.0+PTX"
pip install git+[https://github.com/pytorch/vision.git](https://github.com/pytorch/vision.git)
pip install setuptools wheel ninja
```

**3. Inject OV-DQUO Base & Install Dependencies**
```bash
git clone [https://github.com/xiaomoguhz/ov-dquo.git](https://github.com/xiaomoguhz/ov-dquo.git) temp_dquo
mv temp_dquo/models ./ && mv temp_dquo/datasets ./ && mv temp_dquo/util ./ && mv temp_dquo/engine.py ./ && mv temp_dquo/main.py ./ && mv temp_dquo/config ./ && mv temp_dquo/custom_tools ./
rm -rf temp_dquo
pip install -r requirements.txt
```

**4. Compile Custom Operations (Detectron2 & Deformable Attention)**
```bash
export TORCH_CUDA_ARCH_LIST="9.0;9.0+PTX"
python -m pip install 'git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)' --no-build-isolation --force-reinstall --no-deps

# Patch legacy C++ code for modern PyTorch
find models/ops/src -type f -exec sed -i 's/\.scalar_type().is_cuda()/.is_cuda()/g' {} +
find models/ops/src -type f -exec sed -i 's/\.type().is_cuda()/.is_cuda()/g' {} +

cd models/ops
sh make.sh
cd ../../
```
## Dataset Preparation & Baseline Evaluation

**1. Export your custom dataset (e.g., GAI20) in COCO JSON format.**

**2. Structure the data exactly as follows:**
- Images: data/coco/val2017/
- Annotations: data/coco/annotations/instances_val2017.json

**3. Download the pre-trained OV-DQUO weights (OVDQUO_RN50x4_COCO.pth) into ckpt/.**

**4. Download the region prompt (region_prompt_R50x4.pth) into pretrained/.**

**5. Run the custom baseline evaluation script:**
```bash
python eval_ovdquo.py
```


