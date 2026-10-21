# MTL-TLV Reproducibility Package

This repository provides the source code, and dataset for the MTL-TLV framework and its ablation models.

---

## 1. Installation and Environment Setup

To ensure reproducibility and minimize software-level nondeterminism, we recommend reproducing the experimental environment using **Python 3.9** and **CUDA 11.8**.

### 1.1 Create a Conda Environment

```bash
conda create -n mtl-tlv python=3.9 -y
conda activate mtl-tlv
```

### 1.2 Install PyTorch

The experiments were conducted using **PyTorch 2.1.1+cu118**.

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### 1.3 Install Other Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Software and Hardware Environment

The exact hardware, software, and deterministic settings used in the experiments are provided in:

```text
environment.txt
```

The main environment is:

```text
Python      : 3.9.25
PyTorch     : 2.1.1+cu118
CUDA        : 11.8
cuDNN       : 8.7.0
GPU         : NVIDIA GeForce RTX 4090 D
```

All required Python packages and their versions are listed in:

```text
requirements.txt
```

---

## 3. Repository Contents

```text
MTL-TLV/
│
├── README.md
├── requirements.txt
├── environment.txt
│
├── MLT-TLV_With_Model_A_G.py
│
├── normal_data.rar
└── void_data.rar
```

---

## 4. Dataset

The dataset is provided in the following compressed archives:

```text
normal_data.rar
void_data.rar
```

After extraction, the following six CSV files are used by the training and verification procedures:

```text
normal_data_1.csv
normal_data_2.csv
void_data_1.csv
void_data_2.csv
void_depth_1.csv
void_depth_2.csv
```

Before running the training script, please extract the two archives and place the six CSV files in the repository root directory.

---

## 5. Complete Training Procedure

To reproduce the complete experimental procedure, including SMAC hyperparameter optimization and sequential training of Models A–H, run:

```bash
python MLT-TLV_With_Model_A_G.py
```

---

## 6. Reproducibility Settings

The experiments were conducted using fixed random seeds and deterministic PyTorch settings.

Detailed information on the hardware environment, software versions, random seeds, CUDA/cuDNN settings, TF32 configuration, and deterministic algorithms is provided in:

```text
environment.txt
```
