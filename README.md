# Project Execution Guide

This project includes training, testing, and ablation scripts for multiclass land-use classification using CNNs and Vision Transformers.

---

## 📁 Script Naming Convention

```
train11.py   # Task 1.1 (ResNet-18)
train12.py   # Task 1.2 (ResNet-18 + SE)
train21.py   # Task 2.1 (DeiT-3 Small)
train22.py   # Task 2.2 (DeiT-3 + DyT)
```

(test_*.py and ablation_*.py follow same naming)

---

## Setup

Make the runner script executable:

```bash
chmod +x run.sh
```

---

## Usage

All execution is controlled via a single script:

```bash
./run.sh <mode>
```

---

## Modes

### 1. Run ALL Training

Runs all training scripts in order:
(1.1 → 1.2 → 2.1 → 2.2)

```bash
./run.sh train_all
```

---

### 2. Run ALL Testing

Runs all test scripts in order:

```bash
./run.sh test_all
```

---

### 3. Run ALL Ablations

Runs all ablation scripts in order:

```bash
./run.sh ablation_all
```

---

### 4. Run Individual Tasks (Train + Test + Ablation)

#### Task 1.1

```bash
./run.sh 11
```

#### Task 1.2

```bash
./run.sh 12
```

#### Task 2.1

```bash
./run.sh 21
```

#### Task 2.2

```bash
./run.sh 22
```

Each command runs:

* Training
* Testing
* Ablation

for that specific task.

---

### 5. Run FULL Pipeline

Runs everything in order (train → test → ablation):

1. Task 1.1
2. Task 1.2
3. Task 2.1
4. Task 2.2

```bash
./run.sh full
```

---

## Output Structure

Each run generates structured outputs in:

```
runs/<task_name>/run_<timestamp>/
```

Contents include:

* `config.json` → experiment configuration
* `metrics.csv` → per-epoch metrics
* `best_model.pth` → saved model
* `plots/` → loss, AUC, and metric graphs

---

## Notes

* All scripts use the same training and evaluation logic.
* Early stopping is based on validation AUC.
* Metrics tracked:

  * Accuracy
  * Macro F1 Score
  * Macro ROC-AUC

---

## Summary

| Command        | Description                 |
| -------------- | --------------------------- |
| `train_all`    | Run all training scripts    |
| `test_all`     | Run all test scripts        |
| `ablation_all` | Run all ablations           |
| `11`           | Run Task 1.1 pipeline       |
| `12`           | Run Task 1.2 pipeline       |
| `21`           | Run Task 2.1 pipeline       |
| `22`           | Run Task 2.2 pipeline       |
| `full`         | Run entire project pipeline |

---

This setup ensures clean experiment tracking, reproducibility, and modular execution.
