import os
import json
import torch
from tqdm import tqdm
from ultralytics import FastSAM

COCO_JSON = "data/coco2017/annotations_trainval2017/annotations/instances_train2017.json"
IMAGE_DIR = "data/coco2017/images/train2017/"
OUTPUT_DIR = "output/coco2017/sam_priors_train2017/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Warming up FastSAM Engine for Pre-computation...")
fastsam_model = FastSAM("FastSAM-x.pt")

def extract_and_save_boxes(img_filename, img_id):
    img_path = os.path.join(IMAGE_DIR, img_filename)
    if not os.path.exists(img_path): return
    
    out_file = os.path.join(OUTPUT_DIR, f"{img_id}.pt")
    if os.path.exists(out_file): return # Skip if already computed
        
    results = fastsam_model(img_path, device='cuda', retina_masks=True, imgsz=1024, conf=0.1, iou=0.6, verbose=False)
    
    if len(results[0].boxes.xyxy) == 0:
        boxes_cxcywh = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    else:
        img_h, img_w = results[0].orig_shape
        boxes_xyxy = results[0].boxes.xyxy
        
        scores = results[0].boxes.conf
        _, indices = scores.sort(descending=True)
        boxes_xyxy = boxes_xyxy[indices][:300] # Top 300 max
        
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0 / img_w
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0 / img_h
        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
        
        boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1).cpu()
    
    torch.save(boxes_cxcywh, out_file)

if __name__ == "__main__":
    print(f"Loading COCO Annotations from {COCO_JSON}")
    with open(COCO_JSON, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data['images']
    print(f"Found {len(images)} images. Starting FastSAM Extraction...")
    
    for img in tqdm(images):
        extract_and_save_boxes(img['file_name'], img['id'])
        
    print("Pre-computation Complete. Priors saved to disk.")