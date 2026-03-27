import os

file_path = "datasets/ov_coco.py"
with open(file_path, "r") as f:
    content = f.read()

target = """    if args.label_version=="standard":
        PATHS = {
            "train": (
                root / "images/train2017",
                root / "annotations_trainval2017/annotations" / f"{mode}_train2017.json",
            ),
            "val": (
                root / "images/val2017",
                root / "annotations_trainval2017/annotations" / f"{mode}_val2017.json",
            ),
        }"""

replacement = """    if args.label_version=="standard":
        PATHS = {
            "train": (
                root / "images/train2017",
                root / "annotations_trainval2017/annotations" / f"{mode}_train2017_base.json",
            ),
            "val": (
                root / "images/val2017",
                root / "annotations_trainval2017/annotations" / f"{mode}_val2017_basetarget.json",
            ),
        }"""

if target in content:
    content = content.replace(target, replacement)
    with open(file_path, "w") as f:
        f.write(content)
    print("[*] Successfully routed ov_coco.py to strict OV splits.")
else:
    print("[!] Target block not found. Check your ov_coco.py paths.")