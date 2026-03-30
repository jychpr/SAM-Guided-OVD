# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.
"""
from pathlib import Path

import torch
import json
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import datasets.transforms as T

class OVCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, img_folder, ann_file, transforms, return_masks, pseudo_box
    ):
        super(OVCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.all_categories = {
            k["id"]: k["name"] for k in self.coco.dataset["categories"]
        }
        # --- THESIS: Textual Concept Expansion for COCO ---
        raw_categories = [self.all_categories[k] for k in sorted(self.all_categories.keys())]
        
        COCO_SEMANTICS = {
            'person': 'a human person standing or sitting', 'bicycle': 'a two-wheeled bicycle vehicle',
            'car': 'a passenger car vehicle on a road', 'motorcycle': 'a motorized two-wheeled motorcycle',
            'airplane': 'a flying airplane in the sky or on a runway', 'bus': 'a large passenger bus vehicle',
            'train': 'a locomotive train on railroad tracks', 'truck': 'a large cargo truck vehicle',
            'boat': 'a boat or ship on water', 'traffic light': 'a traffic light signal',
            'fire hydrant': 'a metal fire hydrant on a sidewalk', 'stop sign': 'a red octagonal stop sign',
            'parking meter': 'a coin-operated parking meter', 'bench': 'a sitting bench in a park or street',
            'bird': 'a flying or perched bird animal', 'cat': 'a domestic feline cat pet',
            'dog': 'a domestic canine dog pet', 'horse': 'a large equine horse animal',
            'sheep': 'a woolly sheep animal', 'cow': 'a large bovine cow animal',
            'elephant': 'a large wild elephant with a trunk', 'bear': 'a large wild bear animal',
            'zebra': 'a wild zebra animal with black and white stripes', 'giraffe': 'a tall wild giraffe animal with a long neck',
            'backpack': 'a wearable backpack bag', 'umbrella': 'a handheld umbrella for rain or sun',
            'handbag': 'a carried handbag or purse', 'tie': 'a necktie worn with a shirt',
            'suitcase': 'a luggage suitcase for travel', 'frisbee': 'a flying plastic frisbee disc',
            'skis': 'a pair of long snow skis', 'snowboard': 'a wide flat snowboard for snow',
            'sports ball': 'a round sports ball for playing games', 'kite': 'a flying kite in the sky',
            'baseball bat': 'a wooden or metal baseball bat', 'baseball glove': 'a leather baseball glove',
            'skateboard': 'a wheeled skateboard for riding', 'surfboard': 'a long surfboard for riding waves',
            'tennis racket': 'a stringed tennis racket', 'bottle': 'a cylindrical liquid container bottle',
            'wine glass': 'a glass goblet for wine', 'cup': 'a drinking cup or mug',
            'fork': 'a metal eating fork', 'knife': 'a sharp cutting knife',
            'spoon': 'a metal eating spoon', 'bowl': 'a round eating bowl',
            'banana': 'a yellow curved banana fruit', 'apple': 'a round red or green apple fruit',
            'sandwich': 'a bread sandwich with fillings', 'orange': 'a round orange citrus fruit',
            'broccoli': 'a green broccoli vegetable', 'carrot': 'a long orange carrot vegetable',
            'hot dog': 'a hot dog sausage in a bun', 'pizza': 'a circular baked pizza with toppings',
            'donut': 'a sweet fried donut pastry', 'cake': 'a baked sweet cake dessert',
            'chair': 'a sitting chair piece of furniture', 'couch': 'a multi-person seating couch or sofa',
            'potted plant': 'a green plant in a pot', 'bed': 'a sleeping bed with pillows and blankets',
            'dining table': 'a table used for eating meals', 'toilet': 'a ceramic bathroom toilet',
            'tv': 'a television screen or monitor', 'laptop': 'a portable laptop computer',
            'mouse': 'a computer mouse peripheral', 'remote': 'a handheld remote control',
            'keyboard': 'a computer typing keyboard', 'cell phone': 'a mobile cellular phone',
            'microwave': 'a microwave oven appliance', 'oven': 'a baking oven appliance',
            'toaster': 'a bread toaster appliance', 'sink': 'a water sink basin',
            'refrigerator': 'a large cold refrigerator appliance', 'book': 'a bound paper reading book',
            'clock': 'a time-telling clock', 'vase': 'a decorative vase for holding flowers',
            'scissors': 'a pair of cutting scissors', 'teddy bear': 'a soft stuffed teddy bear toy',
            'hair drier': 'a handheld hair drier appliance', 'toothbrush': 'a bristled toothbrush for cleaning teeth'
        }
        
        self.category_list = [COCO_SEMANTICS.get(name, f"a visual of a {name} object") for name in raw_categories]
        # ---------------------------------------------------------
        self.category_ids = {v: k for k, v in self.all_categories.items()}
        self.label2catid = {
            k: self.category_ids[raw_categories[k]] for k, v in enumerate(self.category_list)
        }
        self.catid2label = {v: k for k, v in self.label2catid.items()}
        self.use_pseudo_box = pseudo_box != ""
        if self.use_pseudo_box:
            with open(pseudo_box, "r") as f:
                pseudo_annotations = json.load(f)
            self.pseudo_annotations = dict()
            for annotation in pseudo_annotations:
                if annotation["image_id"] not in self.pseudo_annotations:
                    self.pseudo_annotations[annotation["image_id"]] = []
                self.pseudo_annotations[annotation["image_id"]].append(annotation)
        self.prepare = ConvertCocoPolysToMask(return_masks, map=self.catid2label)


    def __getitem__(self, idx):
        img, target = super(OVCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        if self.use_pseudo_box:
            pseudo_annotations = (
                self.pseudo_annotations[image_id]
                if image_id in self.pseudo_annotations
                else [])
            target.extend(pseudo_annotations)
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        # --- THESIS: Inject FastSAM Priors into DataLoader (Train + Val) ---
        import os
        prior_val = f"output/coco2017/sam_priors_val2017/{image_id}.pt"
        prior_train = f"output/coco2017/sam_priors_train2017/{image_id}.pt"
        
        if os.path.exists(prior_val):
            target['sam_proposals'] = torch.load(prior_val)
        elif os.path.exists(prior_train):
            target['sam_proposals'] = torch.load(prior_train)
        else:
            target['sam_proposals'] = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        # -----------------------------------------------------
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, map=None):
        self.return_masks = return_masks
        self.map = map

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # classes = [obj["category_id"] for obj in anno]
        classes = [
            obj["category_id"] if "category_id" in obj else -i - 1
            for i, obj in enumerate(anno)
        ]
        pseudo_label_map = {}
        pseudo_mask=[]
        weight=[]
        for i, obj in enumerate(anno):
            if "class_label" in obj:
                pseudo_label_map[-i - 1] = obj["class_label"]
            if "pseudo" in obj:
                pseudo_mask.append(obj["pseudo"])
            else:
                pseudo_mask.append(0) # For compatibility, 0 is added by default to indicate that it is not a pseudo label

            if "weight" in obj:
                weight.append(obj["weight"]) # foreground estimation
            else:
                weight.append(1.0)
        classes = torch.tensor(classes, dtype=torch.int64)
        pseudo_mask= torch.tensor(pseudo_mask, dtype=torch.int64)
        weight= torch.tensor(weight, dtype=torch.float32)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        pseudo_mask=pseudo_mask[keep]
        weight=weight[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["pseudo_label_map"] = pseudo_label_map
        target["pseudo_mask"] = pseudo_mask
        target["weight"] = weight
        if self.map is not None: 
            for idx, label in enumerate(target["labels"]):
                target["labels"][idx] = (
                    self.map[label.item()] if label.item() >= 0 else label.item()
                )
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        # iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = area[keep]
        # target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, args):
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    normalize = T.Compose([T.ToRGB(), T.ToTensor(), T.Normalize(MEAN, STD)])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )
    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )
    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    if args.label_version=="standard":
        PATHS = {
            "train": (
                root / "images/train2017",
                root / "annotations_trainval2017/annotations" / f"{mode}_train2017_base.json",
            ),
            "val": (
                root / "images/val2017",
                root / "annotations_trainval2017/annotations" / f"{mode}_val2017_basetarget.json",
            ),
        }
    elif args.label_version == "custom":
            PATHS["train"]=(
            root / "Images/train2017",
            root / "instances_train2017_base_RN50relabel_pseudo.json",
        )
    img_folder, ann_file = PATHS[image_set]
    dataset = OVCocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_masks=args.masks,
        pseudo_box=args.pseudo_box,
    )
    return dataset
