import os
import json
import torch
from tqdm import tqdm
import cv2
import torchvision.transforms.functional as TF
from torchvision.ops import box_convert
import argparse
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Import OV-DQUO specific modules
from util.slconfig import SLConfig
from main import build_model_main
from util.misc import nested_tensor_from_tensor_list

# --- CONFIGURATION ---
CONFIG_FILE = "config/OV_COCO/OVDQUO_RN50x4.py"
WEIGHTS_FILE = "ckpt/OVDQUO_RN50x4_COCO.pth"
COCO_JSON = "data/gai19coco/test/_annotations.coco.json"
IMAGE_DIR = "data/gai19coco/test/"

# Engineered Prompts
# PROMPT_MAP = {
#     'crimpers': 'crimpers', 'cutter': 'cutter', 'drill': 'drill',
#     'hammer': 'hammer', 'hand file': 'hand file', 'measurement tape': 'measurement tape',
#     'pen': 'pen', 'pliers': 'pliers', 'power supply': 'power supply',
#     'scissors': 'scissors', 'screwdriver': 'screwdriver', 'screws': 'screws',
#     'tape': 'tape', 'tweezers': 'tweezers', 'usb cable': 'usb cable',
#     'vernier caliper': 'vernier caliper', 'whiteboard marker': 'whiteboard marker',
#     'wire': 'wire', 'wrench': 'wrench'
# }
PROMPT_MAP = {
    'crimpers': 'crimping tool',
    'cutter': 'wire cutter',
    'drill': 'power drill',
    'hammer': 'hammer',
    'hand file': 'hand file tool',
    'measurement tape': 'measurement tape',
    'pen': 'pen',
    'pliers': 'pliers',
    'power supply': 'bench power supply',
    'scissors': 'scissors',
    'screwdriver': 'screwdriver',
    'screws': 'metal screws',
    'tape': 'adhesive tape roll',
    'tweezers': 'tweezers',
    'usb cable': 'usb cable',
    'vernier caliper': 'vernier caliper',
    'whiteboard marker': 'whiteboard marker',
    'wire': 'electrical wire',
    'wrench': 'wrench'
}

def run_eval():
    print("Loading Configuration...")
    args = argparse.Namespace()
    cfg = SLConfig.fromfile(CONFIG_FILE)
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
        
    args.device = "cuda"
    args.eval = True
    args.analysis = False
    
    # CRITICAL FIX: Bypass the academic benchmark scaling
    args.target_class_factor = 1.0

    print("Building OV-DQUO Model (RN50x4)...")
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=False)

    # Extract raw model state dict
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(args.device)

    print("Loading COCO JSON...")
    with open(COCO_JSON, 'r') as f:
        coco_data = json.load(f)
        
    # Map JSON IDs to our exact index
    coco_cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    target_names = list(PROMPT_MAP.keys())
    target_name_to_id = {name: idx for idx, name in enumerate(target_names)}
    categories_list = list(PROMPT_MAP.values())

    images_dict = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    preds, targets = [], []

    print(f"Starting Evaluation on {len(images_dict)} images...")
    with torch.no_grad():
        for img_id, img_info in tqdm(images_dict.items()):
            file_name = img_info['file_name']
            img_path = os.path.join(IMAGE_DIR, file_name)
            
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                continue
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image_rgb.shape

            # 1. Parse Ground Truth
            gt_boxes, gt_labels = [], []
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    cat_name = coco_cat_id_to_name.get(ann['category_id'])
                    if cat_name in target_name_to_id:
                        x, y, w, h = ann['bbox'] # COCO is top-left x, y, width, height
                        gt_boxes.append([x, y, x+w, y+h]) # Convert to absolute xyxy
                        gt_labels.append(target_name_to_id[cat_name])
            
            if gt_boxes:
                targets.append({
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "labels": torch.tensor(gt_labels, dtype=torch.int64)
                })
            else:
                targets.append({"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)})

            # 2. Image Preprocessing 
            # CRITICAL FIX: Normalize on CPU first to avoid torchvision GPU kernel crash
            image_tensor = TF.to_tensor(image_rgb)
            image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image_tensor = image_tensor.to(args.device)
            samples = nested_tensor_from_tensor_list([image_tensor])

            # 3. Model Forward Pass
            outputs = model(samples, categories=categories_list.copy())
            
            # 4. Post-Processing
            scores = outputs["pred_logits"][0].sigmoid() # [num_queries, num_classes]
            boxes_norm = outputs["pred_boxes"][0] # [num_queries, 4] (cxcywh normalized)
            
            # Convert to Absolute XYXY
            boxes_abs = boxes_norm * torch.tensor([img_w, img_h, img_w, img_h], device=args.device)
            boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")
            
            # Filter low confidence
            max_scores, labels = scores.max(dim=1)
            keep = max_scores > 0.1
            
            if keep.sum() > 0:
                preds.append({
                    "boxes": boxes_xyxy[keep].cpu(),
                    "scores": max_scores[keep].cpu(),
                    "labels": labels[keep].cpu()
                })
            else:
                preds.append({"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.int64)})

    print("\nComputing metrics. This may take a moment...")
    metric.update(preds, targets)
    results = metric.compute()
    
    print("\n=== OV-DQUO (RN50x4) RESULTS ===")
    print(f"mAP (IoU=0.50:0.95): {results['map'].item():.4f}")
    print(f"mAP@0.50:            {results['map_50'].item():.4f}")
    print(f"mAP@0.75:            {results['map_75'].item():.4f}")

if __name__ == "__main__":
    run_eval()