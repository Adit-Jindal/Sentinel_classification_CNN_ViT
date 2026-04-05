#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Run Management
# -----------------------------
import os
import json
from datetime import datetime

def create_run_dir(base="runs/task1_2_ablation"):
    os.makedirs(base, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{run_id}")
    os.makedirs(run_dir)
    os.makedirs(os.path.join(run_dir, "plots"))
    return run_dir

run_dir = create_run_dir()
print("Saving outputs to:", run_dir)


# -----------------------------
# Load Data
# -----------------------------
from utils import load_data

train_path = "train.csv"
val_path = "validation.csv"

train_data = load_data(train_path)
val_data = load_data(val_path)


# -----------------------------
# Download Pretrained Weights
# -----------------------------
import urllib.request
import ssl
import certifi

url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
ctx = ssl.create_default_context(cafile=certifi.where())

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(req, context=ctx) as response:
    data = response.read()
    with open("resnet18-f37072fd.pth", "wb") as f:
        f.write(data)

print("Downloaded weights!")


# -----------------------------
# SE Block
# -----------------------------
import torch.nn as nn
import torch

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y


# -----------------------------
# Model
# -----------------------------
import torchvision.models as models

class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = models.resnet18(pretrained=False)

        state_dict = torch.load("resnet18-f37072fd.pth")
        self.backbone.load_state_dict(state_dict)

        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.se1(x)

        x = self.backbone.layer2(x)
        x = self.se2(x)

        x = self.backbone.layer3(x)
        x = self.se3(x)

        x = self.backbone.layer4(x)
        x = self.se4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x


# -----------------------------
# Focal Loss
# -----------------------------
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# -----------------------------
# Setup
# -----------------------------
import torch.optim as optim
import params

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------
# Save Config
# -----------------------------
config = {
    "experiment": "ResNet18 + SE Focal Loss Ablation",
    "gamma_values": [1, 2, 3],
    "epochs_per_gamma": 3,
    "lr": params.lr
}

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)


# -----------------------------
# Ablation Loop
# -----------------------------
import time
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np

gamma_results = []
gamma_values = [1, 2, 3]

for gamma in gamma_values:
    criterion = FocalLoss(gamma=gamma)
    print(f"\nTraining with gamma: {gamma}")

    model = ResNet18_SE(num_classes=10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_auc = 0
    best_model_state = None

    for epoch in range(3):
        start = time.time()
        model.train()

        print(f"Running epoch: {epoch+1}")

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_data:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        y_true = label_binarize(all_labels, classes=list(range(10)))
        auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')

        print(f"Epoch {epoch+1} done, Val AUC: {auc:.4f}")
        print(f"Time taken: {time.time()-start:.2f}s")

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()

        gamma_results.append(best_auc)


# -----------------------------
# Plot
# -----------------------------
import matplotlib.pyplot as plt

gamma_results_plot = gamma_results[2::3]

plt.figure()
plt.plot(gamma_values, gamma_results_plot, marker='o')
plt.xlabel("Gamma")
plt.ylabel("Best Val ROC-AUC")
plt.title("Focal Loss Gamma Tuning (SE)")
plt.savefig(os.path.join(run_dir, "plots/ablation_gamma.png"))
plt.close()


# -----------------------------
# Save Ablation Results
# -----------------------------
import pandas as pd

df = pd.DataFrame({
    "gamma": gamma_values,
    "best_val_auc": gamma_results_plot
})

df.to_csv(os.path.join(run_dir, "ablation_results.csv"), index=False)


# -----------------------------
# Test Best Model
# -----------------------------
test_path = "test.csv"
test_data = load_data(test_path)

model = ResNet18_SE(num_classes=10)
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_data:
        images = images.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

all_probs = np.concatenate(all_probs)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)


# -----------------------------
# Test Metrics
# -----------------------------
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

y_true = label_binarize(all_labels, classes=list(range(10)))
auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')

print("Accuracy:", acc)
print("F1 Score:", f1)
print("Macro ROC-AUC:", auc)


# -----------------------------
# Save Test Results
# -----------------------------
results = {
    "accuracy": acc,
    "f1_score": f1,
    "macro_roc_auc": auc
}

with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

pd.DataFrame([results]).to_csv(os.path.join(run_dir, "test_metrics.csv"), index=False)


print("All outputs saved to:", run_dir)