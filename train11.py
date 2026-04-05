#!/usr/bin/env python
# coding: utf-8

# In[16]:


from utils import load_data

train_path = "train.csv"
val_path = "validation.csv"

train_data = load_data(train_path)
val_data = load_data(val_path)


# In[17]:


import urllib.request
import ssl
import certifi

url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
ctx = ssl.create_default_context(cafile=certifi.where())

req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0"}  # pretend to be a browser
)

with urllib.request.urlopen(req, context=ctx) as response:
    data = response.read()
    with open("resnet18-f37072fd.pth", "wb") as f:
        f.write(data)

print("Downloaded weights!")


# In[18]:


import torchvision.models as models
import torch

state_dict = torch.load("resnet18-f37072fd.pth")
model = models.resnet18()
model.load_state_dict(state_dict)


model.fc = torch.nn.Linear(model.fc.in_features, 10) # 10 classes


# In[2]:


import torch
import torch.optim as optim
import torch.nn as nn
import params

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params.lr)


# In[ ]:


from sklearn.metrics import roc_auc_score
import numpy as np

best_auc = 0
best_model_state = None


# In[ ]:


outfile = open("results.txt", 'a')
outfile.write("----------------------------\n")
outfile.write("Training for task 1.1\n")
outfile.write("---------------------\n")


# In[ ]:


import time
from params import num_epochs

losses = []
auc_vals = []

for epoch in range(num_epochs):
  start = time.time()
  model.train()
  train_loss = 0

  outfile.write(f"Running epoch: {epoch+1}")

  for images, labels in train_data:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  train_loss /= len(train_data)

  model.eval()
  all_labels = []
  all_probs = []

  outfile.write(f"  Evaluations for the epoch")

  with torch.no_grad():
    for images, labels in val_data:
      images = images.to(device)
      outputs = model(images)
      probs = torch.softmax(outputs, dim=1)

      all_probs.append(probs.cpu().numpy())
      all_labels.append(labels.numpy())

  all_probs = np.concatenate(all_probs)
  all_labels = np.concatenate(all_labels)

  from sklearn.preprocessing import label_binarize
  y_true = label_binarize(all_labels, classes=list(range(10)))

  auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')
  auc_vals.append(auc)
  losses.append(train_loss)

  outfile.write(f"Epoch {epoch+1} done, Val AUC: {auc:.4f}")
  outfile.write(f"Time taken: {time.time()-start:.2f}s")

  if auc > best_auc:
    best_auc = auc
    best_model_state = model.state_dict()


# In[ ]:


outfile.close()


# In[ ]:


import matplotlib.pyplot as plt

x = list(range(1, 11))

plt.plot(x, losses, marker='o', label='Training Losses')
plt.plot(x, auc_vals, marker='s', label='AUC scores')

plt.xlabel("Epochs")
plt.ylabel("Values")
plt.title("Comparison Plot for standard ResNet-18")
plt.legend()

plt.show()


# In[ ]:


torch.save(best_model_state, "best_model11.pth")


# In[ ]:


with open("results.txt", "a") as f:
    f.write("\n------------------------------\n")
    f.write("Train results for 1.1\n")
    f.write("------------------------------\n")
    f.write("Losses:\n")
    for ls in losses:
        f.write(f"{ls}\n")
    f.write("AUC values:\n")
    for auc in auc_vals:
        f.write(f"{auc}\n")

