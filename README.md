# MTL-TLV Reproducibility Package

This repository provides the source code, dataset, complete training log, trained model weights, and independent verification script for the MTL-TLV framework and its ablation models.

The repository is intended to facilitate transparent reproduction and independent verification of the experimental results.

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
├── void_data.rar
│
├── logs/
│   └── ablation_A_H_complete_training.log
│
└── Weights_and_Verification/
    ├── evaluate_A_H_from_weights.py
    ├── Model_A_best.pth
    ├── Model_B_best.pth
    ├── Model_C_best.pth
    ├── Model_D_best.pth
    ├── Model_E_best.pth
    ├── Model_F_best.pth
    ├── Model_G_best.pth
    └── Model_H_best.pth
```

---

## 4. Ablation Models

The repository contains eight ablation models with the following configurations:

* **Model A:** Temporal single-stream baseline (classification + regression)
* **Model B:** Spectral single-stream baseline (classification + regression)
* **Model C:** Dual-stream naive concatenation, classification only
* **Model D:** Dual-stream naive concatenation, regression only
* **Model E:** Dual-stream naive multi-task integration
* **Model F:** Bidirectional fusion only, without TAFR
* **Model G:** TAFR only, without bidirectional fusion
* **Model H:** Proposed MTL-TLV with bidirectional fusion and TAFR

---

## 5. Dataset

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

Before running the training or verification scripts, please extract the two archives and place the six CSV files in the repository root directory.

The training and verification procedures use the same data preprocessing, FFT transformation, label processing, and fixed data split.

---

## 6. Direct Verification Using the Released Weights

The reported experimental metrics can be independently verified using the released trained weights **without retraining the models**.

The verification script and eight trained checkpoints are provided in:

```text
Weights_and_Verification/
```

Run the verification script from the repository root directory:

```bash
python Weights_and_Verification/evaluate_A_H_from_weights.py
```

The script sequentially loads:

```text
Model_A_best.pth
Model_B_best.pth
Model_C_best.pth
Model_D_best.pth
Model_E_best.pth
Model_F_best.pth
Model_G_best.pth
Model_H_best.pth
```

For each model, the verification procedure:

1. reconstructs the corresponding network architecture;
2. loads the released trained model parameters;
3. reconstructs the same preprocessing and evaluation split used during training;
4. performs model inference;
5. independently recomputes the evaluation metrics.

The independently recomputed metrics include:

* Recognition Accuracy
* Precision
* Recall
* Regression MAE

The recomputed metrics are compared with the corresponding records stored in each checkpoint.

A successful verification for an individual model is reported as:

```text
Checkpoint verification: PASS
```

If all eight checkpoints are successfully verified, the final output is:

```text
OVERALL CHECK: PASS
```

**Important:** the metrics stored in the checkpoint files are not directly used as the reproduced results. All verification metrics are independently recomputed from model predictions after loading the released trained weights. The checkpoint-stored metrics are used only for consistency checking.

---

## 7. Complete Training Procedure

To reproduce the complete experimental procedure, including SMAC hyperparameter optimization and sequential training of Models A–H, run:

```bash
python MLT-TLV_With_Model_A_G.py
```

The complete training record is provided in:

```text
logs/ablation_A_H_complete_training.log
```

---

## 8. Trained Model Weights

The trained model weights are provided in:

```text
Weights_and_Verification/
```

The eight released checkpoints are:

```text
Model_A_best.pth
Model_B_best.pth
Model_C_best.pth
Model_D_best.pth
Model_E_best.pth
Model_F_best.pth
Model_G_best.pth
Model_H_best.pth
```

Each checkpoint contains:

```text
model_id
epoch
model_state_dict
metrics
```

where:

* `model_id` identifies the corresponding ablation model;
* `epoch` records the selected training epoch;
* `model_state_dict` contains the trained neural-network parameters;
* `metrics` contains the evaluation metrics associated with the selected checkpoint.

---

## 9. Reproducibility Settings

The experiments were conducted using fixed random seeds and deterministic PyTorch settings.

Detailed information on the hardware environment, software versions, random seeds, CUDA/cuDNN settings, TF32 configuration, and deterministic algorithms is provided in:

```text
environment.txt
```

The deterministic settings used during the actual training process are also recorded in:

```text
logs/ablation_A_H_complete_training.log
```

---

## 10. Recommended Verification Procedure

For users who only wish to verify the reported numerical results, **model retraining is not required**.

First, extract:

```text
normal_data.rar
void_data.rar
```

and place the six extracted CSV files in the Weights_and_Verification/ directory.

Then run:

```bash
python Weights_and_Verification/evaluate_A_H_from_weights.py
```

The script directly loads the eight released checkpoints and independently recomputes the evaluation metrics from model inference.

For full reproduction beginning with SMAC hyperparameter optimization and model training, run:

```bash
python MLT-TLV_With_Model_A_G.py
```
