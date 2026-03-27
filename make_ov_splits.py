import json
import os

train_in = "data/coco2017/annotations_trainval2017/annotations/instances_train2017.json"
val_in = "data/coco2017/annotations_trainval2017/annotations/instances_val2017.json"

train_out = "data/coco2017/annotations_trainval2017/annotations/instances_train2017_base.json"
val_out = "data/coco2017/annotations_trainval2017/annotations/instances_val2017_basetarget.json"

base_catids = [70, 2, 53, 7, 73, 57, 4, 79, 62, 74, 9, 38, 20, 19, 54, 85, 72, 27, 80, 51, 78, 15, 84, 55, 16, 59, 48, 34, 23, 86, 90, 50, 25, 31, 56, 82, 75, 42, 3, 65, 52, 60, 35, 1, 8, 44, 33, 24]
target_catids = [28, 21, 47, 6, 76, 41, 18, 63, 32, 36, 81, 22, 61, 87, 5, 17, 49]
all_benchmark_ids = base_catids + target_catids

print("Fixing strict OV-COCO training split (48 Base Classes)...")
with open(train_in, 'r') as f:
    train_data = json.load(f)
train_data['annotations'] = [a for a in train_data['annotations'] if a['category_id'] in base_catids]
# THESIS FIX: Filter the metadata header so the dataloader doesn't build an 80-class mapping!
train_data['categories'] = [c for c in train_data['categories'] if c['id'] in base_catids]
with open(train_out, 'w') as f:
    json.dump(train_data, f)

print("Fixing strict OV-COCO validation split (65 Base+Target Classes)...")
with open(val_in, 'r') as f:
    val_data = json.load(f)
val_data['annotations'] = [a for a in val_data['annotations'] if a['category_id'] in all_benchmark_ids]
# THESIS FIX: Filter the metadata header!
val_data['categories'] = [c for c in val_data['categories'] if c['id'] in all_benchmark_ids]
with open(val_out, 'w') as f:
    json.dump(val_data, f)

print("[*] Successfully fully sanitized OV-COCO splits.")