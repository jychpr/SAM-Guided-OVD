import os
import torch
import json
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from models import build_model_main
import argparse
from torchmetrics.detection.mean_ap import MeanAveragePrecision

CONFIG_FILE = 'config/OV_COCO/OVDQUO_RN50x4.py'
WEIGHTS_FILE = 'ckpt/OVDQUO_RN50x4_COCO.pth'
COCO_JSON = 'data/coco/annotations/instances_val2017.json'
IMAGE_DIR = 'data/coco/val2017/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=CONFIG_FILE)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--resume', default=WEIGHTS_FILE)
    parser.add_argument('--num_feature_levels', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--position_embedding', default='sine')
    parser.add_argument('--masks', action='store_true', default=False)
    args, _ = parser.parse_known_args()
    return args

def run_eval():
    print("Loading Configuration and Model...")
    args = get_args()
    
    from util.slconfig import DictAction, SLConfig
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        setattr(args, k, v)
        
    model, criterion, postprocessors = build_model_main(args)
    model.to(args.device)
    
    print(f"Loading Weights from {WEIGHTS_FILE}...")
    checkpoint = torch.load(WEIGHTS_FILE, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    print("Loading COCO JSON...")
    with open(COCO_JSON, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data['images']
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    print(f"Starting Evaluation on {len(images)} images...")
    
    with torch.no_grad():
        for img_info in tqdm(images):
            img_path = os.path.join(IMAGE_DIR, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(img).to(args.device)
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            # TODO: Add specific OV-DQUO forward pass and bounding box extraction here to feed into metric.update()
            
    print("Baseline script executed successfully. Awaiting forward pass implementation.")

if __name__ == '__main__':
    run_eval()