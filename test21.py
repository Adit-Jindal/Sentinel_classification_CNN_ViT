#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import load_data

test_path = "test.csv"
test_data = load_data(test_path)


# In[2]:


import torch
import torch.nn as nn
import timm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = timm.create_model('deit3_small_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, 10)

# Load trained weights
state_dict = torch.load("best_model21.pth", map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()


# In[3]:


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


# In[4]:


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


# In[5]:


print("\n===== Test Results (DeiT) =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Macro F1  : {f1:.4f}")
print(f"Macro AUC : {auc:.4f}")

with open("results.txt", "a") as f:
    f.write("\n------------------------------\n")
    f.write("Test results for 2.1\n")
    f.write("------------------------------\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Macro ROC-AUC: {auc}\n")

