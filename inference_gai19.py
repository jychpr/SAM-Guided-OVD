import os
import json
import torch
import cv2
import warnings
import argparse
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.ops import box_convert
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import FastSAM

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

from util.slconfig import SLConfig
from main import build_model_main
from util.misc import nested_tensor_from_tensor_list

CONFIG_FILE = "config/OV_COCO/OVDQUO_RN50.py"
WEIGHTS_FILE = "output/gai19_finetuned_rn50/checkpoint0099.pth"
COCO_JSON = "data/gai19coco/test/_annotations.coco.json"
IMAGE_DIR = "data/gai19coco/test/"
OUTPUT_DIR = "output/gai19_finetuned_rn50_inference/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

print("Warming up FastSAM Engine...")
fastsam_model = FastSAM("FastSAM-x.pt")

def extract_fastsam_boxes_for_eval(image_path, top_k=300):
    results = fastsam_model(image_path, device='cuda', retina_masks=True, imgsz=1024, conf=0.1, iou=0.6, verbose=False)
    img_h, img_w = results[0].orig_shape
    boxes_xyxy = results[0].boxes.xyxy
    
    if len(boxes_xyxy) == 0:
        return torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        
    scores = results[0].boxes.conf
    _, indices = scores.sort(descending=True)
    boxes_xyxy = boxes_xyxy[indices][:top_k]
    
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0 / img_w
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0 / img_h
    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
    
    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1).cpu()
    return boxes_cxcywh

def run_inference():
    args = argparse.Namespace()
    cfg = SLConfig.fromfile(CONFIG_FILE)
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
        
    args.device = "cuda"
    args.eval = True
    args.analysis = False
    args.target_class_factor = 1.0

    print("Building Modified OV-DQUO Model (RN50)...")
    model, _, _ = build_model_main(args)
    checkpoint = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval().to(args.device)

    with open(COCO_JSON, 'r') as f:
        coco_data = json.load(f)
        
    # --- THE CRITICAL FIX: SORT BY JSON ID EXACTLY LIKE THE DATALOADER ---
    raw_categories = coco_data['categories']
    sorted_categories = sorted(raw_categories, key=lambda x: x['id'])
    
    short_names = [cat['name'] for cat in sorted_categories]
    semantic_definitions = [PROMPT_MAP.get(name, name) for name in short_names]
    json_id_to_tensor_idx = {cat['id']: idx for idx, cat in enumerate(sorted_categories)}
    # ---------------------------------------------------------------------

    images_dict = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    preds, targets = [], []

    print(f"Starting Inference on {len(images_dict)} images...")
    with torch.no_grad():
        for img_id, img_info in tqdm(images_dict.items()):
            img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                continue
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image_rgb.shape

            # 1. Parse Ground Truth using the sorted mapping
            gt_boxes, gt_labels = [], []
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    tensor_idx = json_id_to_tensor_idx.get(ann['category_id'])
                    if tensor_idx is not None:
                        x, y, w, h = ann['bbox'] 
                        gt_boxes.append([x, y, x+w, y+h]) 
                        gt_labels.append(tensor_idx)
            
            if gt_boxes:
                targets.append({"boxes": torch.tensor(gt_boxes, dtype=torch.float32), "labels": torch.tensor(gt_labels, dtype=torch.int64)})
            else:
                targets.append({"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)})

            # 2. Get FastSAM Priors
            sam_boxes = extract_fastsam_boxes_for_eval(img_path, top_k=model.num_queries)
            sam_boxes = sam_boxes.unsqueeze(0).to(args.device)

            # 3. Image Preprocessing
            image_tensor = TF.to_tensor(image_rgb)
            image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image_tensor = image_tensor.to(args.device)
            samples = nested_tensor_from_tensor_list([image_tensor])

            # 4. Forward Pass (Feeding the sorted long semantics)
            outputs = model(samples, categories=semantic_definitions.copy(), sam_proposals=sam_boxes)
            
            # 5. Post-Processing
            scores = outputs["pred_logits"][0].sigmoid() 
            boxes_norm = outputs["pred_boxes"][0] 
            boxes_abs = boxes_norm * torch.tensor([img_w, img_h, img_w, img_h], device=args.device)
            boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")
            
            max_scores, labels = scores.max(dim=1)
            
            eval_keep = max_scores > 0.05
            if eval_keep.sum() > 0:
                preds.append({"boxes": boxes_xyxy[eval_keep].cpu(), "scores": max_scores[eval_keep].cpu(), "labels": labels[eval_keep].cpu()})
            else:
                preds.append({"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.int64)})

            # 6. Visualization (NMS inherently handled by DETR matching, but let's threshold slightly higher)
            vis_keep = max_scores > 0.25
            vis_boxes = boxes_xyxy[vis_keep].cpu().numpy()
            vis_scores = max_scores[vis_keep].cpu().numpy()
            vis_labels = labels[vis_keep].cpu().numpy()

            draw_img = image_cv.copy()
            for box, score, label_idx in zip(vis_boxes, vis_scores, vis_labels):
                if label_idx >= len(short_names):
                    continue # Safety catch for weird DETR background classes
                x1, y1, x2, y2 = map(int, box)
                class_name = short_names[label_idx]
                text = f"{class_name}: {score:.2f}"
                
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_img, text, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out_path = os.path.join(OUTPUT_DIR, img_info['file_name'])
            cv2.imwrite(out_path, draw_img)

    print("\nComputing metrics. This may take a moment...")
    metric.update(preds, targets)
    results = metric.compute()
    
    print("\n=== THESIS ARCHITECTURE GAI19 RESULTS ===")
    print(f"mAP (IoU=0.50:0.95): {results['map'].item():.4f}")
    print(f"mAP@0.50:            {results['map_50'].item():.4f}")
    print(f"mAP@0.75:            {results['map_75'].item():.4f}")

if __name__ == "__main__":
    run_inference()