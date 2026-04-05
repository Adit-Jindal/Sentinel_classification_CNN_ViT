#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Run Management
# -----------------------------
import os
import json
from datetime import datetime

def create_run_dir(base="runs/task2_1"):
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
# Model (DeiT-3 Small)
# -----------------------------
import torch
import torch.nn as nn
import timm

model = timm.create_model('deit3_small_patch16_224', pretrained=True)

# Replace classification head
model.head = nn.Linear(model.head.in_features, 10)


# -----------------------------
# Training Setup
# -----------------------------
import torch.optim as optim
import params

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params.lr)


# -----------------------------
# Save Config
# -----------------------------
from params import num_epochs

config = {
    "model": "deit3_small",
    "lr": params.lr,
    "epochs": num_epochs,
    "batch_size": 32
}

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)


# -----------------------------
# Training Loop
# -----------------------------
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize
import time

best_auc = 0
best_model_state = None

metrics = {
    "epoch": [],
    "train_loss": [],
    "val_auc": [],
    "val_acc": [],
    "val_f1": []
}

for epoch in range(num_epochs):
    start = time.time()
    model.train()
    train_loss = 0

    print(f"Running epoch: {epoch+1}")

    # ---- Training ----
    for images, labels in train_data:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_data)

    # ---- Validation ----
    model.eval()
    all_labels = []
    all_probs = []

    print("  Evaluating...")

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

    # Extra metrics (added only)
    from sklearn.metrics import accuracy_score, f1_score
    preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='macro')

    # Store
    metrics["epoch"].append(epoch+1)
    metrics["train_loss"].append(train_loss)
    metrics["val_auc"].append(auc)
    metrics["val_acc"].append(acc)
    metrics["val_f1"].append(f1)

    print(f"Epoch {epoch+1} done, Loss: {train_loss:.4f}, AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
    print(f"Time taken: {time.time() - start:.2f}s")

    # ---- Save best model ----
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()


# -----------------------------
# Save Model
# -----------------------------
torch.save(best_model_state, os.path.join(run_dir, "best_model.pth"))

print("Training complete. Best AUC:", best_auc)


# -----------------------------
# Save Metrics
# -----------------------------
import pandas as pd

df = pd.DataFrame(metrics)
df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)


# -----------------------------
# Plotting
# -----------------------------
import matplotlib.pyplot as plt

# Loss
plt.figure()
plt.plot(metrics["epoch"], metrics["train_loss"], marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss (DeiT)")
plt.savefig(os.path.join(run_dir, "plots/loss.png"))
plt.close()

# AUC
plt.figure()
plt.plot(metrics["epoch"], metrics["val_auc"], marker='s')
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.title("Validation AUC (DeiT)")
plt.savefig(os.path.join(run_dir, "plots/auc.png"))
plt.close()

# Accuracy & F1
plt.figure()
plt.plot(metrics["epoch"], metrics["val_acc"], label="Accuracy")
plt.plot(metrics["epoch"], metrics["val_f1"], label="F1")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.title("Validation Metrics (DeiT)")
plt.savefig(os.path.join(run_dir, "plots/f1_acc.png"))
plt.close()


print("All outputs saved to:", run_dir)