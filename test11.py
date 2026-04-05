#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Run Management
# -----------------------------
import os
import json
from datetime import datetime

def create_run_dir(base="runs/task1_1_test"):
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
# Model (ResNet-18)
# -----------------------------
import torchvision.models as models
import torch.nn as nn
import torch

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("best_model11.pth", map_location='cpu'))
model.eval()


# -----------------------------
# Inference
# -----------------------------
all_preds = []
all_labels = []
all_probs = []

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

with torch.no_grad():
    for images, labels in test_data:
        images = images.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

import numpy as np
all_probs = np.concatenate(all_probs)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)


# -----------------------------
# Metrics
# -----------------------------
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

y_true = label_binarize(all_labels, classes=list(range(10)))
auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')

print("Accuracy: ", acc)
print("F1 Score: ", f1)
print("Macro ROC-AUC: ", auc)


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


# Optional CSV (for consistency with training logs)
import pandas as pd

df = pd.DataFrame([results])
df.to_csv(os.path.join(run_dir, "test_metrics.csv"), index=False)


print("Test results saved to:", run_dir)