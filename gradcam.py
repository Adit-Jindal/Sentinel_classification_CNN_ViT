#!/usr/bin/env python
# coding: utf-8

# =============================
# CONFIG (EDIT THESE)
# =============================
MODEL_11_PATH = "best_model11.pth"
MODEL_12_PATH = "best_model12.pth"

TEST_CSV = "test.csv"

SE_MODEL_IMPORT = "train12"

# =============================
# IMPORTS
# =============================
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

from utils import load_data

# Dynamic import for SE model
import importlib
se_module = importlib.import_module(SE_MODEL_IMPORT)
ResNet18_SE = getattr(se_module, "ResNet18_SE")

from datetime import datetime

def create_run_dir(base="runs/gradcam"):
        os.makedirs(base, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, f"run_{run_id}")
        os.makedirs(run_dir)
        return run_dir

OUTPUT_DIR = create_run_dir()

# =============================
# Grad-CAM CLASS
# =============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx=None):
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


# =============================
# UTILS
# =============================
def overlay(image, cam):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = 0.5 * heatmap + 0.5 * image
    return overlay.astype(np.uint8)


def tensor_to_image(tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


# =============================
# MODEL LOADERS
# =============================
def load_resnet18(path, device):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_resnet18_se(path, device):
    model = ResNet18_SE()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# =============================
# MAIN
# =============================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    test_data = load_data(TEST_CSV)

    # Load models
    model_11 = load_resnet18(MODEL_11_PATH, device)
    model_12 = load_resnet18_se(MODEL_12_PATH, device)

    # Grad-CAM instances
    cam_11 = GradCAM(model_11, model_11.layer4[-1])
    cam_12 = GradCAM(model_12, model_12.backbone.layer4[-1])

    # Select one image per class
    class_samples = {}

    for images, labels in test_data:
        for i in range(len(labels)):
            label = labels[i].item()
            if label not in class_samples:
                class_samples[label] = images[i].unsqueeze(0)
        if len(class_samples) == 10:
            break

    # Output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate Grad-CAMs
    for cls, img in class_samples.items():
        print(f"Processing class {cls}")

        img = img.to(device)

        cam1 = cam_11.generate(img)
        cam2 = cam_12.generate(img)

        img_np = tensor_to_image(img)

        overlay1 = overlay(img_np, cam1)
        overlay2 = overlay(img_np, cam2)

        # Save images
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"class_{cls}_original.png"), img_np)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"class_{cls}_resnet.png"), overlay1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"class_{cls}_se.png"), overlay2)

    print("\nGrad-CAM generation complete.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()