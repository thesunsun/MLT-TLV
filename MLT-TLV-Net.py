import numpy as np
import torch
import os
import math
from scipy.fftpack import fft
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from ConfigSpace import ConfigurationSpace, Categorical
from smac import HyperparameterOptimizationFacade, Scenario



# Utility functions and datasets
def shuffle_data(processed_data, fft_data, labels, depth):
    indices = np.random.permutation(len(processed_data))
    return (
        processed_data[indices],
        fft_data[indices],
        labels[indices],
        depth[indices]
    )


class GPRDataset(Dataset):
    def __init__(self, raw_data, fft_data, labels, depths):
        self.raw_data = torch.FloatTensor(raw_data).unsqueeze(1)
        self.fft_data = torch.FloatTensor(fft_data).unsqueeze(1)

        # Max-Abs 标准化
        raw_max = torch.max(torch.abs(self.raw_data), dim=2, keepdim=True)[0] + 1e-7
        self.raw_data = self.raw_data / raw_max

        fft_max = torch.max(torch.abs(self.fft_data), dim=2, keepdim=True)[0] + 1e-7
        self.fft_data = self.fft_data / fft_max

        self.labels = torch.LongTensor(labels)
        self.depths = torch.FloatTensor(np.nan_to_num(depths, nan=0))
        self.valid_depths = ~torch.isnan(torch.FloatTensor(depths))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return {
            'raw': self.raw_data[idx],
            'fft': self.fft_data[idx],
            'label': self.labels[idx],
            'depth': self.depths[idx],
            'valid_depth': self.valid_depths[idx]
        }


# Basic network structure
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, block=BasicBlock1D, layers=[2, 2, 2, 2], num_classes=512):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Cross-modal bidirectional fusion strategy
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = math.sqrt(dim)

    def forward(self, q_feat, k_feat, v_feat):
        q = self.query_proj(q_feat).unsqueeze(1)
        k = self.key_proj(k_feat).unsqueeze(1)
        v = self.value_proj(v_feat).unsqueeze(1)

        attn_weights = self.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale)
        out = torch.bmm(attn_weights, v).squeeze(1) + q_feat
        return out


# Task-aware feature routing mechanism
class TaskSpecificRouter(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.routing_gate = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        routing_weights = self.routing_gate(x)
        return x * routing_weights


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_net = ResNet1D()
        self.fft_net = ResNet1D()

        self.cross_attn_time = CrossAttention(dim=512)
        self.cross_attn_freq = CrossAttention(dim=512)

        fused_dim = 1024

        self.cls_router = TaskSpecificRouter(channel=fused_dim)
        self.reg_router = TaskSpecificRouter(channel=fused_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, raw, fft):
        raw_feat = self.raw_net(raw)
        fft_feat = self.fft_net(fft)

        cross_raw = self.cross_attn_time(q_feat=raw_feat, k_feat=fft_feat, v_feat=fft_feat)
        cross_fft = self.cross_attn_freq(q_feat=fft_feat, k_feat=raw_feat, v_feat=raw_feat)

        shared_fused = torch.cat([cross_raw, cross_fft], dim=1)

        cls_feat = self.cls_router(shared_fused)
        reg_feat = self.reg_router(shared_fused)

        out_cls = self.classifier(cls_feat)
        out_reg = self.regressor(reg_feat)

        return out_cls, out_reg


# Loss function and training control
class HybridLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mae = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, outputs, targets):
        cls_pred, reg_pred = outputs
        cls_true, depth_true, valid_mask = targets

        cls_loss = self.ce(cls_pred, cls_true)

        if valid_mask.sum() > 0:
            reg_loss = self.mae(reg_pred[valid_mask], depth_true[valid_mask].unsqueeze(1))
        else:
            reg_loss = torch.tensor(0.0, device=cls_loss.device)

        total_loss = cls_loss + self.loss_weight * reg_loss
        return total_loss, cls_loss, reg_loss


# Data reading and running
def ReadData(file_name1, file_name2, file_name3, file_name4, file_name5, file_name6):
    normal_1 = pd.read_csv(file_name1).iloc[:, 1:]
    normal_2 = pd.read_csv(file_name2).iloc[:, 1:]
    normal_data = pd.concat([normal_1, normal_2], axis=1)

    void_1 = pd.read_csv(file_name3).iloc[:, 1:]
    void_2 = pd.read_csv(file_name4).iloc[:, 1:]
    void_data = pd.concat([void_1, void_2], axis=1)

    void_Depth_1 = pd.read_csv(file_name5).iloc[:, 1]
    void_Depth_2 = pd.read_csv(file_name6).iloc[:, 1]
    void_Depth = pd.concat([void_Depth_1.to_frame(), void_Depth_2.to_frame()], axis=0)

    groups = np.concatenate([
        np.zeros(normal_1.shape[1], dtype=int),
        np.ones(normal_2.shape[1], dtype=int),
        np.zeros(void_1.shape[1], dtype=int),
        np.ones(void_2.shape[1], dtype=int)
    ])

    AllData_r = pd.concat([normal_data, void_data], axis=1)
    AllData = AllData_r.T

    return AllData, void_Depth, groups


def fft_transform(FS, N, RawData):
    DataSpeed_Initial = fft(RawData, N)
    DataSpeed_Abs = abs(DataSpeed_Initial) / N / math.sqrt(2)
    DataSpeed_Abs[0] = 0
    DataSpeed_Abs = DataSpeed_Abs[0:50]
    DataSpeed_FFt = 2 * DataSpeed_Abs
    return DataSpeed_FFt


# SMAC optimization function
def run_smac_optimization(processed_data, fft_data, labels, depth, groups):
    print("\n>>> Start running SMAC (Objective: Minimize the combined loss on the validation set) <<<")

    # 获取数据分割
    total_samples = len(processed_data)
    val_size_absolute = int(total_samples * 0.1)
    idx_line1 = np.where(groups == 0)[0]
    idx_line2 = np.where(groups == 1)[0]

    train_l1_part, val_idx = train_test_split(
        idx_line1, test_size=val_size_absolute, random_state=42, stratify=labels[idx_line1]
    )
    train_idx = np.concatenate([idx_line2, train_l1_part])

    # Set the total number of trials
    total_trials = 10
    trial_count = [0]

    # SMAC evaluation function
    def evaluate_config(config, seed: int) -> float:
        trial_count[0] += 1
        print(f"\n[SMAC Progress: {trial_count[0]}/{total_trials}] Start evaluating the configuration...")
        print(f" -> Try parameters: LR={config['learning_rate']}, WD={config['weight_decay']}, Balancing Weights={config['loss_weight']}")

        model = FusionNet().to(DEVICE)
        device = next(model.parameters()).device
        optimizer = optim.Adam(model.parameters(),
                               lr=config["learning_rate"],
                               weight_decay=config["weight_decay"])
        criterion = HybridLoss(loss_weight=config["loss_weight"])

        train_set = GPRDataset(processed_data[train_idx], fft_data[train_idx], labels[train_idx], depth[train_idx])
        val_set = GPRDataset(processed_data[val_idx], fft_data[val_idx], labels[val_idx], depth[val_idx])

        train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=512)

        best_val_loss = float('inf')
        smac_epochs = 70

        for epoch in range(1, smac_epochs + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                raw_input, fft_input = batch['raw'].to(DEVICE), batch['fft'].to(DEVICE)
                label_target, depth_target, valid_mask = batch['label'].to(DEVICE), batch['depth'].to(DEVICE), batch[
                    'valid_depth'].to(DEVICE)

                cls_out, reg_out = model(raw_input, fft_input)
                loss, _, _ = criterion((cls_out, reg_out), (label_target, depth_target, valid_mask))
                loss.backward()
                optimizer.step()
            model.eval()
            val_mae = 0.0
            valid_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    raw_input, fft_input = batch['raw'].to(DEVICE), batch['fft'].to(DEVICE)
                    depth_target = batch['depth'].to(DEVICE)
                    valid_mask = batch['valid_depth'].to(DEVICE)

                    cls_out, reg_out = model(raw_input, fft_input)

                    if valid_mask.sum() > 0:
                        val_mae += F.l1_loss(reg_out[valid_mask],
                                             depth_target[valid_mask].unsqueeze(1)).item() * valid_mask.sum().item()
                        valid_samples += valid_mask.sum().item()

            avg_val_mae = val_mae / (valid_samples + 1e-7)
            if avg_val_mae < best_val_loss:
                best_val_loss = avg_val_mae

            if epoch % 10 == 0 or epoch == smac_epochs:
                print(f"    - Epoch {epoch:02d}/{smac_epochs}")

        return best_val_loss


    # 1. Configuration Space
    cs = ConfigurationSpace()
    lr = Categorical("learning_rate", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    wd = Categorical("weight_decay", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    lw = Categorical("loss_weight", [1e-2, 1e-1, 1.0, 10.0])
    cs.add_hyperparameters([lr, wd, lw])

    # 2. run SMAC
    scenario = Scenario(cs, deterministic=True, n_trials=total_trials)
    smac = HyperparameterOptimizationFacade(scenario, evaluate_config)
    incumbent = smac.optimize()

    print(f"\n======================================================")
    print(f" The optimal hyperparameters are:")
    print(f" Learning Rate: {incumbent['learning_rate']}")
    print(f" Weight Decay: {incumbent['weight_decay']}")
    print(f" Loss Weight: {incumbent['loss_weight']}")
    print(f"======================================================\n")

    return incumbent


# Model evaluation and visualization
def evaluate_and_plot(model, val_loader, device, model_path="best_model.pth"):
    print("\n>>> Loading the best model for final evaluation and plotting...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_cls_trues = []
    all_cls_preds = []

    all_reg_trues = []
    all_reg_preds = []

    with torch.no_grad():
        for batch in val_loader:
            raw_input = batch['raw'].to(device)
            fft_input = batch['fft'].to(device)
            label_target = batch['label'].to(device)
            depth_target = batch['depth'].to(device)
            valid_mask = batch['valid_depth'].to(device)

            cls_out, reg_out = model(raw_input, fft_input)

            # Collect and classify prediction results
            pred_cls = cls_out.argmax(dim=1)
            all_cls_preds.extend(pred_cls.cpu().numpy())
            all_cls_trues.extend(label_target.cpu().numpy())

            # Collect effective regression prediction results
            if valid_mask.sum() > 0:
                all_reg_preds.extend(reg_out[valid_mask].cpu().numpy().flatten())
                all_reg_trues.extend(depth_target[valid_mask].cpu().numpy().flatten())

        # Draw and save the confusion matrix
    cm = confusion_matrix(all_cls_trues, all_cls_preds)
    accuracy = np.trace(cm) / np.sum(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal (0)', 'Void (1)'],
                yticklabels=['Normal (0)', 'Void (1)'])

    plt.title(f'Classification Confusion Matrix (Accuracy: {accuracy:.2%})')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

    # Draw a curve comparing regression true values with predicted values
    if len(all_reg_trues) > 0:
        # To make the line chart more intuitive, we sort the actual values and arrange the predicted values accordingly
        sorted_indices = np.argsort(all_reg_trues)
        sorted_true = np.array(all_reg_trues)[sorted_indices]
        sorted_pred = np.array(all_reg_preds)[sorted_indices]

        mae = mean_absolute_error(sorted_true, sorted_pred)

        plt.figure(figsize=(10, 5))
        plt.plot(sorted_true, label='True Depth', color='black', linewidth=2)
        plt.plot(sorted_pred, label='Predicted Depth', color='red', alpha=0.7, linestyle='--')
        plt.title(f'Depth Regression: True vs Predicted Curve (MAE: {mae:.3f})')
        plt.xlabel('Sample Index (Sorted by True Depth)')
        plt.ylabel('Depth Value')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig('regression_curve.png', dpi=300)
        plt.show()
        print(f"\n The charts have been saved separately as 'confusion_matrix.png' and 'regression_curve.png'")


# Strict 9:1 data partitioning
def custom_domain_adaptation_train(processed_data, fft_data, labels, depth, groups, best_config):
    results = []
    chooseMAE = 1000
    best_acc = 0.0
    total_samples = len(processed_data)
    val_size_absolute = int(total_samples * 0.1)

    idx_line1 = np.where(groups == 0)[0]
    idx_line2 = np.where(groups == 1)[0]

    folds_setup = []

    train_l1_part, val_fold2 = train_test_split(
        idx_line1,
        test_size=val_size_absolute,
        random_state=42,
        stratify=labels[idx_line1]
    )
    train_fold2 = np.concatenate([idx_line2, train_l1_part])
    folds_setup.append((train_fold2, val_fold2, 1))

    for fold, (train_idx, val_idx, val_target_line) in enumerate(folds_setup):
        reg_Dataframe = pd.DataFrame()
        print(f'\n======================================================')
        print(f'Start final training of the model (Apply the optimal parameters of SMAC)')
        print(f'Total data: {total_samples}')
        print(f'Training set proportion 90%: {len(train_idx)} traces | Validation set proportion 10%: {len(val_idx)} traces')
        print(f'======================================================')

        model = FusionNet().to(DEVICE)

        # Apply SMAC
        optimizer = optim.Adam(model.parameters(),
                               lr=best_config["learning_rate"],
                               weight_decay=best_config["weight_decay"])
        criterion = HybridLoss(loss_weight=best_config["loss_weight"])

        train_set = GPRDataset(processed_data[train_idx], fft_data[train_idx], labels[train_idx], depth[train_idx])
        val_set = GPRDataset(processed_data[val_idx], fft_data[val_idx], labels[val_idx], depth[val_idx])

        train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=512)

        for epoch in range(1, 101):
            model.train()
            total_loss = total_cls_loss = total_reg_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                raw_input = batch['raw'].to(DEVICE)
                fft_input = batch['fft'].to(DEVICE)
                label_target = batch['label'].to(DEVICE)
                depth_target = batch['depth'].to(DEVICE)
                valid_mask = batch['valid_depth'].to(DEVICE)

                cls_out, reg_out = model(raw_input, fft_input)
                loss, cls_loss, reg_loss = criterion((cls_out, reg_out), (label_target, depth_target, valid_mask))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()

            model.eval()
            val_correct = 0
            val_mae = 0.0
            valid_samples = 0
            val_total_loss = val_cls_loss_sum = val_reg_loss_sum = 0.0
            PreDepth = []
            TrueDepth = []

            with torch.no_grad():
                for batch in val_loader:
                    raw_input = batch['raw'].to(DEVICE)
                    fft_input = batch['fft'].to(DEVICE)
                    label_target = batch['label'].to(DEVICE)
                    depth_target = batch['depth'].to(DEVICE)
                    valid_mask = batch['valid_depth'].to(DEVICE)

                    cls_out, reg_out = model(raw_input, fft_input)

                    loss, cls_loss, reg_loss = criterion((cls_out, reg_out), (label_target, depth_target, valid_mask))
                    val_total_loss += loss.item()
                    val_cls_loss_sum += cls_loss.item()
                    val_reg_loss_sum += reg_loss.item()

                    pred = cls_out.argmax(dim=1)
                    val_correct += (pred == label_target).sum().item()

                    PreDepth = PreDepth + reg_out[valid_mask].cpu().numpy().flatten().tolist()
                    TrueDepth = TrueDepth + depth_target[valid_mask].unsqueeze(1).cpu().numpy().flatten().tolist()

                    if valid_mask.sum() > 0:
                        val_mae += F.l1_loss(reg_out[valid_mask],
                                             depth_target[valid_mask].unsqueeze(1)).item() * valid_mask.sum().item()
                        valid_samples += valid_mask.sum().item()

            val_acc = val_correct / len(val_set)
            val_mae = val_mae / (valid_samples + 1e-7)

            if chooseMAE > val_mae:
                torch.save(model.state_dict(), "best_model.pth")
                chooseMAE = val_mae
                best_acc = val_acc

            avg_train_loss = total_loss / len(train_loader)
            # avg_train_cls = total_cls_loss / len(train_loader)
            # avg_train_reg = total_reg_loss / len(train_loader)
            avg_val_loss = val_total_loss / len(val_loader)
            # avg_val_cls = val_cls_loss_sum / len(val_loader)
            # avg_val_reg = val_reg_loss_sum / len(val_loader)

            print(f'Epoch {epoch:02d} | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f}  | '
                  f'Val Acc: {val_acc:.4f} | '
                  f'Val MAE: {val_mae:.4f}')

        results.append(val_acc)
        evaluate_and_plot(model, val_loader, DEVICE, model_path="best_model.pth")
    print(f'\n======================================================')
    print(f' -> Optimal classification accuracy: {best_acc:.2%}')
    print(f' -> Optimal depth regression error (MAE): {chooseMAE:.4f}')
    print(f'======================================================')


if __name__ == "__main__":
    np.random.seed(42)
    N = 512
    FS = 20480
    BATCH_SIZE = 512
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data, depths, groups = ReadData('normal_data_1.csv', 'normal_data_2.csv', 'void_data_1.csv', 'void_data_2.csv',
                                        'void_depth_1.csv', 'void_depth_2.csv', )

    depths = depths.to_numpy().T.reshape(-1)
    nodepths = np.ones(6159)
    nodepths[:6159] = np.nan
    depth = np.concatenate([nodepths, depths])
    labels = np.concatenate([np.zeros(6159, dtype=int), np.ones(4995, dtype=int)])
    # Human experts annotate noise
    noise_rate = 0.012
    n_samples = len(labels)
    noise_indices = np.random.choice(n_samples, int(noise_rate * n_samples), replace=False)
    labels[noise_indices] = 1 - labels[noise_indices]

    train_fft = pd.DataFrame()
    for i in range(raw_data.shape[0]):
        fftData = fft_transform(FS, N, raw_data.iloc[i].tolist())
        train_fft = pd.concat([train_fft, pd.Series(fftData)], axis=1)
    train_fft = train_fft.T

    processed_data = raw_data.to_numpy().reshape(11154, 512)
    fft_data = train_fft.to_numpy().reshape(11154, 50)
    # Use SMAC to find the optimal hyperparameters
    best_config = run_smac_optimization(processed_data, fft_data, labels, depth, groups)

    # Incorporate the optimal hyperparameters into the existing training logic
    custom_domain_adaptation_train(processed_data, fft_data, labels, depth, groups, best_config)
