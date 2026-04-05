#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import load_data

test_path = "test.csv"

test_data = load_data(test_path)


# In[3]:


import torch.nn as nn
import torch

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.size()

        # Squeeze
        y = x.mean(dim=(2, 3))  # Global Avg Pool

        # Excitation
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        y = y.view(b, c, 1, 1)

        # Scale
        return x * y


# In[12]:


import torchvision.models as models

class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        # load pretrained weights
        # state_dict = torch.load("best_model12.pth", map_location="cpu")

        # SE blocks
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # model.load_state_dict(state_dict)
        # replace fc
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


# In[13]:


import torchvision.models as models
import torch.nn as nn
import torch

model = ResNet18_SE(num_classes=10)
state_dict = torch.load("best_model12.pth", map_location='cpu')
model.load_state_dict(state_dict)


# In[14]:


all_preds = []
all_labels = []
all_probs = []

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

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


# In[15]:


from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

y_true = label_binarize(all_labels, classes=list(range(10)))
auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')

acc = accuracy_score(all_labels, all_preds)

f1 = f1_score(all_labels, all_preds, average='macro')

print("Accuracy: ", acc)
print("F1 Score: ", f1)
print("Macro ROC-AUC: ", auc)

with open("results.txt", "a") as f:
    f.write("\n------------------------------\n")
    f.write("Test results for 1.2\n")
    f.write("------------------------------\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Macro ROC-AUC: {auc}\n")

