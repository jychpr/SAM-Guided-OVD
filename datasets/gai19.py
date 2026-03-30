
import os
import torch
from datasets.coco import make_coco_transforms, CocoDetection

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

class GAI19Dataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, sam_priors_dir):
        super().__init__(img_folder, ann_file, transforms, return_masks)
        self.sam_priors_dir = sam_priors_dir
        
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        self.category_list = [PROMPT_MAP.get(cat['name'], cat['name']) for cat in cats]

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        
        image_id = self.ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        base_name = os.path.splitext(file_name)[0]
        pt_path = os.path.join(self.sam_priors_dir, f"{base_name}.pt")
        
        if os.path.exists(pt_path):
            sam_priors = torch.load(pt_path)
        else:
            sam_priors = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
            
        target['sam_proposals'] = sam_priors
        
        # --- THE PHANTOM LIMB FIXES ---
        num_boxes = len(target['labels']) if 'labels' in target else 0
        
        # 1. pseudo_mask: Tell DTQT that all boxes are real Ground Truth (False)
        target['pseudo_mask'] = torch.zeros(num_boxes, dtype=torch.bool)
        
        # 2. weight: Tell DTQT that all boxes have 100% confidence (1.0)
        target['weight'] = torch.ones(num_boxes, dtype=torch.float32)
        
        return img, target

def build_gai19(image_set, args):
    if image_set == 'train':
        img_dir = "data/gai19coco/train/"
        ann_file = "data/gai19coco/train/_annotations.coco.json"
        sam_dir = "output/gai19coco/sam_priors_train/"
    else:
        img_dir = "data/gai19coco/test/"
        ann_file = "data/gai19coco/test/_annotations.coco.json"
        sam_dir = "output/gai19coco/sam_priors_test/"
        
    dataset = GAI19Dataset(img_dir, ann_file, transforms=make_coco_transforms(image_set), return_masks=False, sam_priors_dir=sam_dir)
    return dataset
