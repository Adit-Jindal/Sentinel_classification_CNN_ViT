#!/usr/bin/env python
# coding: utf-8

# In[5]:


from utils import load_data

train_path = "train.csv"
val_path = "validation.csv"

train_data = load_data(train_path)
val_data = load_data(val_path)


# In[ ]:


import torch.nn as nn
import torch 

class DyT(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))  # learnable

    def forward(self, x):
        return torch.tanh(self.alpha * x)


# In[ ]:


def replace_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, DyT())
        else:
            replace_layernorm(child)


# In[ ]:


import torch
import torch.nn as nn
import timm

model = timm.create_model('deit3_small_patch16_224', pretrained=True)
replace_layernorm(model)

# Replace classification head
model.head = nn.Linear(model.head.in_features, 10)


# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # can be None or tensor of class weights

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# In[8]:


import torch.optim as optim
import params

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = model.to(device)

criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=params.lr)


# In[ ]:


outfile = open("results.txt", 'a')
outfile.write("----------------------------\n")
outfile.write("Ablation for task 2.2\n")
outfile.write("---------------------\n")


# In[ ]:


import time
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18
import torch.nn as nn
import numpy as np
import params 

gamma_results = []
gamma_values = [1,2,3]

for gamma in gamma_values:
  criterion = FocalLoss(gamma=gamma)
  outfile.write(f"Training with gamma: {gamma}\n")

  model = timm.create_model('deit3_small_patch16_224', pretrained=True)
  model.head = nn.Linear(model.head.in_features, 10)
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)


  best_auc = 0
  best_model_state = None


  for epoch in range(3): # shorter run to judge trends
    start = time.time()
    model.train()

    outfile.write(f"Running epoch: {epoch+1}\n")

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

    outfile.write(f"  Evaluations for the epoch\n")

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

    outfile.write(f"Epoch {epoch+1} done, Val AUC: {auc:.4f}\n")
    outfile.write(f"Time taken: {time.time()-start:.2f}s\n")

    if auc > best_auc:
      best_auc = auc
      best_model_state = model.state_dict()

    gamma_results.append(best_auc)


# In[ ]:


outfile.close()


# In[ ]:


import matplotlib.pyplot as plt

gamma_results = gamma_results[2::3]
plt.plot(gamma_values, gamma_results, marker='o')
plt.xlabel("Gamma")
plt.ylabel("Best Val ROC-AUC")
plt.title("Focal Loss Gamma Tuning")
plt.show()
plt.savefig("Ablation_plot22")


# In[ ]:


with open("results.txt", "a") as f:
    f.write("\n------------------------------\n")
    f.write("Ablation training results for 2.2\n")
    f.write("------------------------------\n")
    f.write("Values of ROC-AUC across changing gamma:\n")
    for gr in gamma_results:
        f.write(f"{gr}\n")


# ------------------------

# In[ ]:


from utils import load_data

test_path = "test.csv"

test_data = load_data(test_path)


# In[ ]:


import torch
import torch.nn as nn
import timm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = timm.create_model('deit3_small_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, 10)

# Load trained weights
# state_dict = torch.load(best_model_state, map_location=device)
model.load_state_dict(best_model_state)

model = model.to(device)
model.eval()


# In[ ]:


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


# In[ ]:


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


# In[ ]:


print("\n===== Test Results (DeiT) =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Macro F1  : {f1:.4f}")
print(f"Macro AUC : {auc:.4f}")

with open("results.txt", "a") as f:
    f.write("\n------------------------------\n")
    f.write("Ablation Test results for 2.2\n")
    f.write("------------------------------\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Macro ROC-AUC: {auc}\n")

