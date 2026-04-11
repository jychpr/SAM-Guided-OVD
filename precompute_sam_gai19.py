import os
import json
import torch
import cv2
from tqdm import tqdm
from ultralytics import FastSAM
import warnings

# Suppress warnings for cleaner terminal
warnings.filterwarnings("ignore", category=FutureWarning)

# --- SPLIT CONFIGURATIONS ---
SPLITS = [
    {
        "name": "Train Split (25 Images)",
        "json": "data/gai19coco/train/_annotations.coco.json",
        "img_dir": "data/gai19coco/train/",
        "out_dir": "output/gai19coco/sam_priors_train/"
    },
    {
        "name": "Test Split (300 Images)",
        "json": "data/gai19coco/test/_annotations.coco.json",
        "img_dir": "data/gai19coco/test/",
        "out_dir": "output/gai19coco/sam_priors_test/"
    }
]

print("Warming up FastSAM Engine...")
model = FastSAM("FastSAM-x.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_splits():
    for split in SPLITS:
        print(f"\n=== Processing {split['name']} ===")
        
        # Ensure output directory exists
        os.makedirs(split['out_dir'], exist_ok=True)
        
        if not os.path.exists(split['json']):
            print(f"Error: Could not find JSON at {split['json']}. Skipping split.")
            continue
            
        with open(split['json'], 'r') as f:
            coco_data = json.load(f)
            
        images = coco_data['images']
        print(f"Found {len(images)} images to process.")
        
        # THESIS FIX: Increase query limit to flood the decoder
        top_k = 800 
        
        for img_info in tqdm(images, desc=f"Extracting Priors"):
            img_name = img_info['file_name']
            img_path = os.path.join(split['img_dir'], img_name)
            
            if not os.path.exists(img_path):
                print(f"\nWarning: Image {img_path} not found. Skipping.")
                continue
                
            # Define output tensor path (.pt)
            base_name = os.path.splitext(img_name)[0]
            out_pt_path = os.path.join(split['out_dir'], f"{base_name}.pt")
            
            # THESIS FIX: Removed the os.path.exists() check here so it FORECES an overwrite of the old, flawed tensors.

            # THESIS FIX: Dropped conf to 0.02 to capture specular metallic edges
            results = model(img_path, device=device, retina_masks=True, imgsz=1024, conf=0.02, iou=0.6, verbose=False)
            img_h, img_w = results[0].orig_shape
            boxes_xyxy = results[0].boxes.xyxy
            
            if len(boxes_xyxy) == 0:
                # Fallback for empty images to prevent crashing
                boxes_cxcywh = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
            else:
                scores = results[0].boxes.conf
                _, indices = scores.sort(descending=True)
                boxes_xyxy = boxes_xyxy[indices][:top_k]
                
                # Convert to normalized [cx, cy, w, h]
                cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0 / img_w
                cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0 / img_h
                w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
                h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
                
                boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1).cpu()
                
            # Save tensor
            torch.save(boxes_cxcywh, out_pt_path)

if __name__ == "__main__":
    process_splits()
    print("\nPre-computation complete. Dense priors safely stored and overwritten in the output/ directory.")