#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Run Management
# -----------------------------
import os
import json
from datetime import datetime

def create_run_dir(base="runs/task2_2_test"):
    os.makedirs(base, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{run_id}")
    os.makedirs(run_dir)
    return run_dir

run_dir = create_run_dir()
print("Saving outputs to:", run_dir)


# -----------------------------
# Load Data
# -----------------------------
from utils import load_data

test_path = "test.csv"
test_data = load_data(test_path)


# -----------------------------
# DyT (unchanged)
# -----------------------------
import torch.nn as nn
import torch 

class DyT(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.tanh(self.alpha * x)


def replace_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, DyT())
        else:
            replace_layernorm(child)


# -----------------------------
# Model (DeiT + DyT)
# -----------------------------
import timm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = timm.create_model('deit3_small_patch16_224', pretrained=False)
replace_layernorm(model)
model.head = nn.Linear(model.head.in_features, 10)

# Load trained weights
state_dict = torch.load("best_model22.pth", map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()


# -----------------------------
# Inference
# -----------------------------
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
import numpy as np

all_labels = []
all_probs = []
all_preds = []

with torch.no_grad():
    for images, labels in test_data:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())


# -----------------------------
# Metrics
# -----------------------------
all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)

# Accuracy
acc = accuracy_score(all_labels, all_preds)

# Macro F1
f1 = f1_score(all_labels, all_preds, average='macro')

# Macro AUC
y_true = label_binarize(all_labels, classes=list(range(10)))
auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')


# -----------------------------
# Print Results
# -----------------------------
print("\n===== Test Results (DeiT + DyT) =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Macro F1  : {f1:.4f}")
print(f"Macro AUC : {auc:.4f}")


# -----------------------------
# Save Results
# -----------------------------
results = {
    "accuracy": acc,
    "f1_score": f1,
    "macro_roc_auc": auc
}

with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)


# Optional CSV
import pandas as pd

df = pd.DataFrame([results])
df.to_csv(os.path.join(run_dir, "test_metrics.csv"), index=False)


print("Test results saved to:", run_dir)