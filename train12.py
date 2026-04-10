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

def main():
    
    import os
    import json
    from datetime import datetime

    def create_run_dir(base="runs/task1_2"):
        os.makedirs(base, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, f"run_{run_id}")
        os.makedirs(run_dir)
        os.makedirs(os.path.join(run_dir, "plots"))
        return run_dir

    run_dir = create_run_dir()
    from utils import load_data

    train_path = "train.csv"
    val_path = "validation.csv"

    train_data = load_data(train_path)
    val_data = load_data(val_path)


   
    import urllib.request
    import ssl
    import certifi

    url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    ctx = ssl.create_default_context(cafile=certifi.where())

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(req, context=ctx) as response:
        data = response.read()
        with open("resnet18-f37072fd.pth", "wb") as f:
            f.write(data)

    print("Downloaded weights!")


    import torch.optim as optim
    import params

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18_SE(num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.lr)


    from params import num_epochs

    config = {
        "model": "resnet18_se",
        "lr": params.lr,
        "epochs": num_epochs,
        "batch_size": 32
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


    import time
    import numpy as np
    from sklearn.metrics import roc_auc_score

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

        print("  Evaluations for the epoch")

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

        # Extra metrics
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
        print(f"Time taken: {time.time()-start:.2f}s")

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()


    torch.save(best_model_state, os.path.join(run_dir, "best_model.pth"))

    import pandas as pd

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    import matplotlib.pyplot as plt

    # Loss
    plt.figure()
    plt.plot(metrics["epoch"], metrics["train_loss"], marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss (ResNet + SE)")
    plt.savefig(os.path.join(run_dir, "plots/loss.png"))
    plt.close()

    # AUC
    plt.figure()
    plt.plot(metrics["epoch"], metrics["val_auc"], marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.title("Validation AUC (ResNet + SE)")
    plt.savefig(os.path.join(run_dir, "plots/auc.png"))
    plt.close()

    # Accuracy & F1
    plt.figure()
    plt.plot(metrics["epoch"], metrics["val_acc"], label="Accuracy")
    plt.plot(metrics["epoch"], metrics["val_f1"], label="F1")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Validation Metrics (ResNet + SE)")
    plt.savefig(os.path.join(run_dir, "plots/f1_acc.png"))
    plt.close()


    print("Training complete. Best AUC:", best_auc)
    print("All outputs saved to:", run_dir)


if __name__=="__main__":
    main()