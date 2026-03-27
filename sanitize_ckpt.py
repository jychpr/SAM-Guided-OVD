import torch

ckpt_path = 'ckpt/OVDQUO_RN50x4_COCO.pth'
out_path = 'ckpt/OVDQUO_RN50x4_COCO_80cls.pth'

print(f"Loading {ckpt_path}...")
# THESIS FIX: Bypass PyTorch 2.6 strict loading to read legacy argparse objects
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

if 'label_enc.weight' in state_dict:
    print("[*] Deleting 48-class label_enc.weight to allow 80-class expansion...")
    del state_dict['label_enc.weight']

torch.save(checkpoint, out_path)
print(f"[*] Saved sanitized checkpoint to {out_path}")