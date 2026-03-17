import os
import json
import torch
from tqdm import tqdm
import cv2
import torchvision.transforms.functional as TF
from torchvision.ops import box_convert
import argparse
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import FastSAM
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Import OV-DQUO specific modules
from util.slconfig import SLConfig
from main import build_model_main
from util.misc import nested_tensor_from_tensor_list

# print("Warming up Original SAM (ViT-H) Engine. This will take VRAM...")
# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# sam.to(device="cuda")
# # Generate dense proposals similar to FastSAM's default behavior
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100, 
# )

CONFIG_FILE = "config/OV_COCO/OVDQUO_RN50x4.py"
WEIGHTS_FILE = "ckpt/OVDQUO_RN50x4_COCO.pth"
COCO_JSON = "data/gai19coco/test/_annotations.coco.json"
IMAGE_DIR = "data/gai19coco/test/"

# PROMPT_MAP = {
#     'crimpers': 'crimpers', 'cutter': 'cutter', 'drill': 'drill',
#     'hammer': 'hammer', 'hand file': 'hand file', 'measurement tape': 'measurement tape',
#     'pen': 'pen', 'pliers': 'pliers', 'power supply': 'power supply',
#     'scissors': 'scissors', 'screwdriver': 'screwdriver', 'screws': 'screws',
#     'tape': 'tape', 'tweezers': 'tweezers', 'usb cable': 'usb cable',
#     'vernier caliper': 'vernier caliper', 'whiteboard marker': 'whiteboard marker',
#     'wire': 'wire', 'wrench': 'wrench'
# }
# PROMPT_MAP = {
#     'crimpers': 'crimping tool', 'cutter': 'wire cutter', 'drill': 'power drill',
#     'hammer': 'hammer', 'hand file': 'hand file tool', 'measurement tape': 'measurement tape',
#     'pen': 'pen', 'pliers': 'pliers', 'power supply': 'bench power supply',
#     'scissors': 'scissors', 'screwdriver': 'screwdriver', 'screws': 'metal screws',
#     'tape': 'adhesive tape roll', 'tweezers': 'tweezers', 'usb cable': 'usb cable',
#     'vernier caliper': 'vernier caliper', 'whiteboard marker': 'whiteboard marker',
#     'wire': 'electrical wire', 'wrench': 'wrench'
# }
PROMPT_MAP = {
    'crimpers': 'a metal crimping tool with handled grips, used for wire crimping',
    'cutter': 'a metal wire cutter plier tool with sharp cutting jaws',
    'drill': 'a handheld electric power drill tool',
    'hammer': 'a heavy metal hammer tool with a handle',
    'hand file': 'a long metal hand file tool for smoothing surfaces',
    'measurement tape': 'a rolled measurement tape ruler',
    'pen': 'a writing pen',
    'pliers': 'a pair of metal pliers hand tool with gripping jaws',
    'power supply': 'a box-shaped bench power supply unit with digital display and knobs',
    'scissors': 'a pair of cutting scissors',
    'screwdriver': 'a handheld screwdriver tool',
    'screws': 'small metal hardware screws',
    'tape': 'a roll of adhesive tape',
    'tweezers': 'a small metal tweezers tool for precision gripping',
    'usb cable': 'a black usb cord cable',
    'vernier caliper': 'a metal vernier caliper measuring tool',
    'whiteboard marker': 'a cylindrical whiteboard marker pen',
    'wire': 'a thin electrical wire cord',
    'wrench': 'a metal wrench hand tool'
}

# Pre-load FastSAM globally so we don't reload it every loop
print("Warming up FastSAM Engine...")
fastsam_model = FastSAM("FastSAM-x.pt")

def extract_fastsam_boxes_for_eval(image_path, top_k=300):
    results = fastsam_model(image_path, device='cuda', retina_masks=True, imgsz=1024, conf=0.1, iou=0.6, verbose=False)
    img_h, img_w = results[0].orig_shape
    boxes_xyxy = results[0].boxes.xyxy
    
    if len(boxes_xyxy) == 0:
        return torch.tensor([[0.5, 0.5, 0.1, 0.1]]) # Small fallback box
        
    scores = results[0].boxes.conf
    _, indices = scores.sort(descending=True)
    boxes_xyxy = boxes_xyxy[indices][:top_k]
    
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0 / img_w
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0 / img_h
    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
    
    # Return exactly N boxes. NO PADDING.
    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1).cpu()
    return boxes_cxcywh

# def extract_heavy_sam_boxes(image_rgb, top_k=300):
#     # Meta SAM expects RGB numpy arrays
#     masks = mask_generator.generate(image_rgb)
    
#     if len(masks) == 0:
#         return torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        
#     # Sort masks by predicted IoU (confidence)
#     masks = sorted(masks, key=(lambda x: x['predicted_iou']), reverse=True)
#     masks = masks[:top_k]
    
#     img_h, img_w, _ = image_rgb.shape
#     boxes_cxcywh = []
    
#     for ann in masks:
#         # Meta SAM returns bbox as [x_min, y_min, width, height] in absolute pixels
#         x, y, w, h = ann['bbox']
        
#         cx = (x + w / 2.0) / img_w
#         cy = (y + h / 2.0) / img_h
#         norm_w = w / img_w
#         norm_h = h / img_h
        
#         boxes_cxcywh.append([cx, cy, norm_w, norm_h])
        
#     return torch.tensor(boxes_cxcywh, dtype=torch.float32)

def run_eval():
    args = argparse.Namespace()
    cfg = SLConfig.fromfile(CONFIG_FILE)
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
        
    args.device = "cuda"
    args.eval = True
    args.analysis = False
    args.target_class_factor = 1.0

    print("Building Modified OV-DQUO Model...")
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval().to(args.device)

    with open(COCO_JSON, 'r') as f:
        coco_data = json.load(f)
        
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

    print(f"Starting Thesis Evaluation on {len(images_dict)} images...")
    with torch.no_grad():
        for img_id, img_info in tqdm(images_dict.items()):
            img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
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
                        x, y, w, h = ann['bbox'] 
                        gt_boxes.append([x, y, x+w, y+h]) 
                        gt_labels.append(target_name_to_id[cat_name])
            
            if gt_boxes:
                targets.append({"boxes": torch.tensor(gt_boxes, dtype=torch.float32), "labels": torch.tensor(gt_labels, dtype=torch.int64)})
            else:
                targets.append({"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)})

            # 2. Get FastSAM Priors
            sam_boxes = extract_fastsam_boxes_for_eval(img_path, top_k=model.num_queries)
            # sam_boxes = extract_heavy_sam_boxes(image_rgb, top_k=model.num_queries)
            sam_boxes = sam_boxes.unsqueeze(0).to(args.device) # Shape: [1, 300, 4]

            # 3. Image Preprocessing
            image_tensor = TF.to_tensor(image_rgb)
            image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image_tensor = image_tensor.to(args.device)
            samples = nested_tensor_from_tensor_list([image_tensor])

            # 4. THESIS FORWARD PASS
            outputs = model(samples, categories=categories_list.copy(), sam_proposals=sam_boxes)
            
            # 5. Post-Processing
            scores = outputs["pred_logits"][0].sigmoid() 
            boxes_norm = outputs["pred_boxes"][0] 
            boxes_abs = boxes_norm * torch.tensor([img_w, img_h, img_w, img_h], device=args.device)
            boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")
            
            max_scores, labels = scores.max(dim=1)
            keep = max_scores > 0.1
            
            if keep.sum() > 0:
                preds.append({"boxes": boxes_xyxy[keep].cpu(), "scores": max_scores[keep].cpu(), "labels": labels[keep].cpu()})
            else:
                preds.append({"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.int64)})

    print("\nComputing metrics. This may take a moment...")
    metric.update(preds, targets)
    results = metric.compute()
    
    print("\n=== THESIS ARCHITECTURE (FastSAM + OV-DQUO) RESULTS ===")
    print(f"mAP (IoU=0.50:0.95): {results['map'].item():.4f}")
    print(f"mAP@0.50:            {results['map_50'].item():.4f}")
    print(f"mAP@0.75:            {results['map_75'].item():.4f}")

if __name__ == "__main__":
    run_eval()