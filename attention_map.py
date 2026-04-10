#!/usr/bin/env python
# coding: utf-8

# =============================
# CONFIG (EDIT THESE)
# =============================
MODEL_PATH = "best_model21.pth"
TEST_CSV = "test.csv"

import os
from datetime import datetime

def create_run_dir(base="runs/attention"):
        os.makedirs(base, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, f"run_{run_id}")
        os.makedirs(run_dir)
        return run_dir

OUTPUT_DIR = create_run_dir()

# =============================
# IMPORTS
# =============================
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm

from utils import load_data

# =============================
# ATTENTION HOOK
# =============================
class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attentions = []

        blk = model.blocks[-1]
        # for blk in model.blocks:
        blk.attn.register_forward_hook(self.get_attention)

    def get_attention(self, module, input, output):
        # Extract attention weights manually
        # input[0] = x → [B, N, C]
        x = input[0]

        B, N, C = x.shape
        qkv = module.qkv(x)
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)

        self.attentions.append(attn.detach())

    def clear(self):
        self.attentions = []

# =============================
# ATTENTION ROLLOUT
# =============================
def compute_rollout(attentions):
    result = None

    for attn in attentions:
        attn = attn.mean(dim=1)  # [B, N, N]

        eye = torch.eye(attn.size(-1)).to(attn.device)
        eye = eye.unsqueeze(0).expand(attn.size(0), -1, -1)

        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn
        else:
            result = torch.bmm(attn, result)

    return result


# =============================
# UTILS
# =============================
def tensor_to_image(tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


def overlay(image, attn_map):
    attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    overlay = 0.5 * heatmap + 0.5 * image
    return overlay.astype(np.uint8)


# =============================
# MAIN
# =============================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = timm.create_model('deit3_small_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 10)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Attach attention extractor
    extractor = AttentionExtractor(model)

    # Load data
    test_data = load_data(TEST_CSV)

    # Pick one image per class
    class_samples = {}
    for images, labels in test_data:
        for i in range(len(labels)):
            label = labels[i].item()
            if label not in class_samples:
                class_samples[label] = images[i].unsqueeze(0)
        if len(class_samples) == 10:
            break

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each class
    for cls, img in class_samples.items():
        print(f"Processing class {cls}")

        extractor.clear()

        img = img.to(device)
        # Forward pass (collect attention)
        _ = model(img)

        # Compute rollout
        attn = compute_rollout(extractor.attentions)

        # Extract CLS attention
        attn_map = attn[:, 0, 1:]  # CLS → patches

        # Reshape to grid
        num_patches = attn_map.shape[-1]
        grid_size = int(np.sqrt(num_patches))  # should be 14 for 224/16

        attn_map = attn_map.reshape(grid_size, grid_size).detach().cpu().numpy()

        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)

        # Convert image
        img_np = tensor_to_image(img)

        # Overlay
        overlay_img = overlay(img_np, attn_map)

        # Save
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"class_{cls}_original.png"), img_np)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"class_{cls}_attention.png"), overlay_img)

    print("\nAttention maps generated successfully.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()