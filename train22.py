#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import load_data

train_path = "train.csv"
val_path = "validation.csv"

train_data = load_data(train_path)
val_data = load_data(val_path)


# In[2]:


import torch.nn as nn
import torch 

class DyT(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))  # learnable

    def forward(self, x):
        return torch.tanh(self.alpha * x)


# In[3]:


def replace_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, DyT())
        else:
            replace_layernorm(child)


# In[4]:


import torch
import torch.nn as nn
import timm

model = timm.create_model('deit3_small_patch16_224', pretrained=True)
replace_layernorm(model)

# Replace classification head
model.head = nn.Linear(model.head.in_features, 10)


# In[5]:


import torch.optim as optim
import params

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params.lr)


# In[6]:


outfile = open("results.txt", 'a')
outfile.write("----------------------------\n")
outfile.write("Training for task 2.2\n")
outfile.write("---------------------\n")


# In[7]:


from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize
import time
from params import num_epochs

best_auc = 0
best_model_state = None

losses = []
auc_vals = []

for epoch in range(num_epochs):
    start = time.time()
    model.train()
    train_loss = 0

    outfile.write(f"Running epoch: {epoch+1}\n")

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

    outfile.write("  Evaluating...\n")

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
    losses.append(train_loss)
    auc_vals.append(auc)

    outfile.write(f"Epoch {epoch+1} done, Val AUC: {auc:.4f}\n")
    outfile.write(f"Time taken: {time.time() - start:.2f}s\n")

    # ---- Save best model ----
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()


# In[8]:


outfile.close()


# In[9]:


import matplotlib.pyplot as plt

x = list(range(1, 11))

plt.plot(x, losses, marker='o', label='Training Losses')
plt.plot(x, auc_vals, marker='s', label='AUC scores')

plt.xlabel("Epochs")
plt.ylabel("Values")
plt.title("Comparison Plot for standard ResNet-18")
plt.legend()
plt.savefig("Train_plot22.png")
plt.show()


# In[10]:


torch.save(best_model_state, "best_model22.pth")

print("Training complete. Best AUC:", best_auc)


# In[11]:


with open("results.txt", "a") as f:
    f.write("\n------------------------------\n")
    f.write("Train results for 2.2\n")
    f.write("------------------------------\n")
    f.write("Losses:\n")
    for ls in losses:
        f.write(f"{ls}\n")
    f.write("AUC values:\n")
    for auc in auc_vals:
        f.write(f"{auc}\n")

