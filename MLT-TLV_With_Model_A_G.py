from __future__ import annotations
import copy
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ConfigSpace import Categorical, ConfigurationSpace
from scipy.fft import fft
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from smac import HyperparameterOptimizationFacade, Scenario
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings('ignore', message='Deterministic behavior was enabled.*', category=UserWarning)
logging.getLogger('smac').setLevel(logging.WARNING)
logging.getLogger('ConfigSpace').setLevel(logging.WARNING)
MODEL_DESCRIPTIONS: Dict[str, str] = {'A': 'Temporal single-stream baseline (classification + regression)',
                                      'B': 'Spectral single-stream baseline (classification + regression)',
                                      'C': 'Dual-stream naive concatenation, classification only',
                                      'D': 'Dual-stream naive concatenation, regression only',
                                      'E': 'Dual-stream naive multi-task integration',
                                      'F': 'Bidirectional fusion only, no TAFR',
                                      'G': 'TAFR only, no bidirectional fusion',
                                      'H': 'Proposed MTL-TLV (bidirectional fusion + TAFR)'}
VALID_MODELS = tuple(MODEL_DESCRIPTIONS)

SETTINGS = {
    "data_dir": ".",
    "output_dir": "MTL-TLV-With-Model-A-G-Output",
    "models": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "epochs": 100,
    "batch_size": 512,
    "seed": 35,
    "split_seed": 35,
    "val_ratio": 0.10,
    "label_noise_rate": 0.012,
    "fft_bins": 50,
    "n_fft": 512,
    "device": "cuda:0",
    "smac_trials": 10,
    "smac_epochs": 100,
    "smac_lr_choices": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "smac_weight_decay_choices": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    "smac_loss_weight_choices": [1e-2, 1e-1, 1.0, 10.0],
}


def _bootstrap_environment() -> None:
    seed = str(SETTINGS['seed'])
    required = {'PYTHONHASHSEED': seed, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    for key, value in required.items():
        os.environ[key] = value
    if os.environ.get('MTL_TLV_ENV_BOOTSTRAPPED') == '1' and os.environ.get('PYTHONHASHSEED') == seed:
        return
    child_env = os.environ.copy()
    child_env.update(required)
    child_env['MTL_TLV_ENV_BOOTSTRAPPED'] = '1'
    script_path = Path(__file__).resolve()
    completed = subprocess.run([sys.executable, str(script_path)], env=child_env, cwd=str(script_path.parent),
                               check=False)
    raise SystemExit(completed.returncode)
_bootstrap_environment()


class GPRDataset(Dataset):
    """GPR dataset with per-trace max-absolute normalization."""

    def __init__(self, raw_data: np.ndarray, fft_data: np.ndarray, labels: np.ndarray, depths: np.ndarray) -> None:
        raw_tensor = torch.as_tensor(raw_data, dtype=torch.float32).unsqueeze(1)
        fft_tensor = torch.as_tensor(fft_data, dtype=torch.float32).unsqueeze(1)
        raw_max = torch.amax(torch.abs(raw_tensor), dim=2, keepdim=True).clamp_min(1e-07)
        fft_max = torch.amax(torch.abs(fft_tensor), dim=2, keepdim=True).clamp_min(1e-07)
        self.raw_data = raw_tensor / raw_max
        self.fft_data = fft_tensor / fft_max
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        depth_tensor = torch.as_tensor(depths, dtype=torch.float32)
        self.valid_depths = ~torch.isnan(depth_tensor)
        self.depths = torch.nan_to_num(depth_tensor, nan=0.0)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'raw': self.raw_data[idx], 'fft': self.fft_data[idx], 'label': self.labels[idx],
                'depth': self.depths[idx], 'valid_depth': self.valid_depths[idx]}


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                    bias=False), nn.BatchNorm1d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet1D(nn.Module):
    """1D ResNet-18-style feature extractor returning a 512-D vector."""

    def __init__(self, block: type[BasicBlock1D]=BasicBlock1D, layers: Sequence[int]=(2, 2, 2, 2),
                 feature_dim: int=512) -> None:
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
        self.fc = nn.Linear(512 * block.expansion, feature_dim)

    def _make_layer(self, block: type[BasicBlock1D], out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        modules: List[nn.Module] = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            modules.append(block(self.in_channels, out_channels))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class CrossAttention(nn.Module):
    """ Cross-modal fusion operation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = math.sqrt(dim)

    def forward(self, q_feat: torch.Tensor, k_feat: torch.Tensor, v_feat: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(q_feat).unsqueeze(1)
        k = self.key_proj(k_feat).unsqueeze(1)
        v = self.value_proj(v_feat).unsqueeze(1)
        attn_weights = self.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale)
        return torch.bmm(attn_weights, v).squeeze(1) + q_feat


class TaskSpecificRouter(nn.Module):
    """Task-aware feature routing (TAFR) module."""
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.routing_gate = nn.Sequential(
            # Stage 1: Information squeeze stage / Eq. (6)
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            # Stage 2: Mask excitation stage / Eq. (7)
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights = self.routing_gate(x)
        # Stage 3: Residual gating and feature decoupling stage / Eq. (8)
        return x + x * routing_weights


class ClassificationHead(nn.Module):

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                 nn.Linear(256, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                 nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AblationModel(nn.Module):
    """Unified implementation of Models A-H."""

    def __init__(self, model_id: str, feature_dim: int=512) -> None:
        super().__init__()
        model_id = model_id.upper()
        if model_id not in VALID_MODELS:
            raise ValueError(f'Unknown model_id={model_id}; choose from {VALID_MODELS}')
        self.model_id = model_id
        self.has_classification = model_id != 'D'
        self.has_regression = model_id != 'C'
        self.use_raw = model_id != 'B'
        self.use_fft = model_id != 'A'
        self.use_fusion = model_id in {'F', 'H'}
        self.use_router = model_id in {'G', 'H'}
        if self.use_raw:
            self.raw_net = ResNet1D(feature_dim=feature_dim)
        if self.use_fft:
            self.fft_net = ResNet1D(feature_dim=feature_dim)
        if self.use_raw and self.use_fft:
            shared_dim = feature_dim * 2
            if self.use_fusion:
                self.cross_attn_time = CrossAttention(dim=feature_dim)
                self.cross_attn_freq = CrossAttention(dim=feature_dim)
        else:
            shared_dim = feature_dim
        if self.use_router:
            if self.has_classification:
                self.cls_router = TaskSpecificRouter(channel=shared_dim)
            if self.has_regression:
                self.reg_router = TaskSpecificRouter(channel=shared_dim)
        if self.has_classification:
            self.classifier = ClassificationHead(shared_dim)
        if self.has_regression:
            self.regressor = RegressionHead(shared_dim)

    def forward(self, raw: torch.Tensor, fft_input: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if self.model_id == 'A':
            shared = self.raw_net(raw)
        elif self.model_id == 'B':
            shared = self.fft_net(fft_input)
        else:
            raw_feat = self.raw_net(raw)
            fft_feat = self.fft_net(fft_input)
            if self.use_fusion:
                cross_raw = self.cross_attn_time(raw_feat, fft_feat, fft_feat)
                cross_fft = self.cross_attn_freq(fft_feat, raw_feat, raw_feat)
                shared = torch.cat([cross_raw, cross_fft], dim=1)
            else:
                shared = torch.cat([raw_feat, fft_feat], dim=1)
        cls_pred: Optional[torch.Tensor] = None
        reg_pred: Optional[torch.Tensor] = None
        if self.has_classification:
            cls_feat = self.cls_router(shared) if self.use_router else shared
            cls_pred = self.classifier(cls_feat)
        if self.has_regression:
            reg_feat = self.reg_router(shared) if self.use_router else shared
            reg_pred = self.regressor(reg_feat)
        return {'cls': cls_pred, 'reg': reg_pred}


class AblationLoss(nn.Module):
    """Task-aware loss supporting classification-only and regression-only models."""

    def __init__(self, loss_weight: float=1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mae = nn.L1Loss()
        self.loss_weight = float(loss_weight)

    def forward(self, outputs: Dict[str, Optional[torch.Tensor]], cls_true: torch.Tensor, depth_true: torch.Tensor,
                valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = cls_true.device
        cls_loss = torch.zeros((), device=device)
        reg_loss = torch.zeros((), device=device)
        if outputs['cls'] is not None:
            cls_loss = self.ce(outputs['cls'], cls_true)
        if outputs['reg'] is not None:
            if bool(valid_mask.any()):
                reg_loss = self.mae(outputs['reg'][valid_mask], depth_true[valid_mask].unsqueeze(1))
            else:
                reg_loss = outputs['reg'].sum() * 0.0
        total_loss = cls_loss + self.loss_weight * reg_loss
        return (total_loss, cls_loss, reg_loss)


def set_reproducibility(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def read_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filenames = {'normal_1': 'normal_data_1.csv', 'normal_2': 'normal_data_2.csv', 'void_1': 'void_data_1.csv',
                 'void_2': 'void_data_2.csv', 'void_depth_1': 'void_depth_1.csv', 'void_depth_2': 'void_depth_2.csv'}
    paths = {key: data_dir / name for key, name in filenames.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing required CSV files:\n  ' + '\n  '.join(missing))
    normal_1 = pd.read_csv(paths['normal_1']).iloc[:, 1:]
    normal_2 = pd.read_csv(paths['normal_2']).iloc[:, 1:]
    void_1 = pd.read_csv(paths['void_1']).iloc[:, 1:]
    void_2 = pd.read_csv(paths['void_2']).iloc[:, 1:]
    normal_data = pd.concat([normal_1, normal_2], axis=1)
    void_data = pd.concat([void_1, void_2], axis=1)
    raw_data = pd.concat([normal_data, void_data], axis=1).T.to_numpy(dtype=np.float32)
    depth_1 = pd.read_csv(paths['void_depth_1']).iloc[:, 1].to_numpy(dtype=np.float32)
    depth_2 = pd.read_csv(paths['void_depth_2']).iloc[:, 1].to_numpy(dtype=np.float32)
    void_depths = np.concatenate([depth_1, depth_2])
    n_normal = normal_data.shape[1]
    n_void = void_data.shape[1]
    if len(void_depths) != n_void:
        raise ValueError(f'Void depth count {len(void_depths)} does not match void trace count {n_void}.')
    labels = np.concatenate([np.zeros(n_normal, dtype=np.int64), np.ones(n_void, dtype=np.int64)])
    depths = np.concatenate([np.full(n_normal, np.nan, dtype=np.float32), void_depths])
    groups = np.concatenate([np.zeros(normal_1.shape[1], dtype=np.int64), np.ones(normal_2.shape[1], dtype=np.int64),
                             np.zeros(void_1.shape[1], dtype=np.int64), np.ones(void_2.shape[1], dtype=np.int64)])
    return (raw_data, labels, depths, groups)


def build_fft_features(raw_data: np.ndarray, n_fft: int, bins: int) -> np.ndarray:
    transformed = fft(raw_data, n=n_fft, axis=1)
    magnitude = np.abs(transformed) / n_fft / math.sqrt(2.0)
    magnitude[:, 0] = 0.0
    return (2.0 * magnitude[:, :bins]).astype(np.float32)


def inject_label_noise(labels: np.ndarray, rate: float, seed: int) -> np.ndarray:
    noisy = labels.copy()
    count = int(len(noisy) * rate)
    if count > 0:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(noisy), count, replace=False)
        noisy[indices] = 1 - noisy[indices]
    return noisy


def make_split(labels: np.ndarray, groups: np.ndarray, val_ratio: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    group0 = np.where(groups == 0)[0]
    group1 = np.where(groups == 1)[0]
    val_size = int(len(labels) * val_ratio)
    if val_size <= 0 or val_size >= len(group0):
        raise ValueError('Invalid validation size for the fixed domain split.')
    train_group0, val_idx = train_test_split(group0, test_size=val_size, random_state=split_seed,
                                             stratify=labels[group0])
    train_idx = np.concatenate([group1, train_group0])
    return (np.sort(train_idx), np.sort(val_idx))


def make_loaders(raw_data: np.ndarray, fft_data: np.ndarray, labels: np.ndarray, depths: np.ndarray, train_idx: np.ndarray,
                 val_idx: np.ndarray, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_set = GPRDataset(raw_data[train_idx], fft_data[train_idx], labels[train_idx], depths[train_idx])
    val_set = GPRDataset(raw_data[val_idx], fft_data[val_idx], labels[val_idx], depths[val_idx])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return (train_loader, val_loader)


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: AblationLoss, device: torch.device) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    cls_true: List[int] = []
    cls_pred: List[int] = []
    reg_true: List[float] = []
    reg_pred: List[float] = []
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            raw = batch['raw'].to(device, non_blocking=True)
            fft_input = batch['fft'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            depths = batch['depth'].to(device, non_blocking=True)
            valid = batch['valid_depth'].to(device, non_blocking=True)
            outputs = model(raw, fft_input)
            loss, _, _ = criterion(outputs, labels, depths, valid)
            total_loss += float(loss.item())
            batch_count += 1
            if outputs['cls'] is not None:
                prediction = outputs['cls'].argmax(dim=1)
                cls_true.extend(labels.cpu().tolist())
                cls_pred.extend(prediction.cpu().tolist())
            if outputs['reg'] is not None and bool(valid.any()):
                reg_true.extend(depths[valid].cpu().numpy().tolist())
                reg_pred.extend(outputs['reg'][valid].squeeze(1).cpu().numpy().tolist())
    accuracy = float(accuracy_score(cls_true, cls_pred)) if cls_true else float('nan')
    mae = float(mean_absolute_error(reg_true, reg_pred)) if reg_true else float('nan')
    metrics = {'loss': total_loss / max(batch_count, 1), 'accuracy': accuracy, 'mae': mae}
    arrays = {'cls_true': np.asarray(cls_true, dtype=np.int64), 'cls_pred': np.asarray(cls_pred, dtype=np.int64),
              'reg_true': np.asarray(reg_true, dtype=np.float32), 'reg_pred': np.asarray(reg_pred, dtype=np.float32)}
    return (metrics, arrays)


def plot_confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal (0)', 'Void (1)'], yticklabels=['Normal (0)', 'Void (1)'],
           ylabel='True label', xlabel='Predicted label', title='Model H confusion matrix')
    threshold = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > threshold else 'black')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_regression_curve(true_depth: np.ndarray, pred_depth: np.ndarray, output_path: Path) -> None:
    order = np.argsort(true_depth)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true_depth[order], label='True depth', linewidth=2)
    ax.plot(pred_depth[order], label='Predicted depth', linestyle='--', alpha=0.75)
    ax.set(title='Model H depth regression', xlabel='Sample index (sorted by true depth)', ylabel='Depth')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_smac_search(raw_data: np.ndarray, fft_data: np.ndarray, labels: np.ndarray, depths: np.ndarray,
                    train_idx: np.ndarray, val_idx: np.ndarray, device: torch.device) -> Dict[str, float]:
    cs = ConfigurationSpace(seed=SETTINGS['seed'])
    cs.add([Categorical('learning_rate', SETTINGS['smac_lr_choices']),
            Categorical('weight_decay', SETTINGS['smac_weight_decay_choices']),
            Categorical('loss_weight', SETTINGS['smac_loss_weight_choices'])])
    trial_counter = 0

    def evaluate_candidate(candidate, seed: int=0) -> float:
        nonlocal trial_counter
        trial_counter += 1
        set_reproducibility(SETTINGS['seed'])
        lr = float(candidate['learning_rate'])
        wd = float(candidate['weight_decay'])
        loss_weight = float(candidate['loss_weight'])
        print(f"SMAC trial {trial_counter}/{SETTINGS['smac_trials']} | LR={lr:g} | WD={wd:g} | loss_weight={loss_weight:g}")
        train_loader, val_loader = make_loaders(raw_data, fft_data, labels, depths, train_idx, val_idx,
                                                SETTINGS['batch_size'], SETTINGS['seed'])
        model = AblationModel("H").to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = AblationLoss(loss_weight)
        best_mae = float('inf')
        best_epoch = 0
        try:
            for epoch in range(1, SETTINGS['smac_epochs'] + 1):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    raw = batch['raw'].to(device, non_blocking=True)
                    fft_input = batch['fft'].to(device, non_blocking=True)
                    batch_labels = batch['label'].to(device, non_blocking=True)
                    batch_depths = batch['depth'].to(device, non_blocking=True)
                    valid = batch['valid_depth'].to(device, non_blocking=True)
                    outputs = model(raw, fft_input)
                    loss, _, _ = criterion(outputs, batch_labels, batch_depths, valid)
                    loss.backward()
                    optimizer.step()
                metrics, _ = evaluate_model(model, val_loader, criterion, device)
                if metrics['mae'] < best_mae:
                    best_mae = metrics['mae']
                    best_epoch = epoch
                print(f"  Epoch {epoch:03d}/{SETTINGS['smac_epochs']} | Val Loss={metrics['loss']:.4f} | "
                      f"Acc={metrics['accuracy'] * 100:.2f}% | MAE={metrics['mae']:.4f}")
            print(f'Best single epoch: {best_epoch} | MAE={best_mae:.4f}')
            return float(best_mae)
        except Exception as exc:
            print(f'[SMAC ERROR] {type(exc).__name__}: {exc}')
            return 1000000000000.0
        finally:
            del model, optimizer, criterion, train_loader, val_loader
            torch.cuda.empty_cache()
    print('Starting SMAC hyperparameter optimization')
    print('=' * 88)
    with tempfile.TemporaryDirectory(prefix='mtl_tlv_smac_') as temp_dir:
        scenario = Scenario(cs, deterministic=True, n_trials=SETTINGS['smac_trials'], seed=SETTINGS['seed'],
                            output_directory=Path(temp_dir), crash_cost=1000000000000.0)
        facade = HyperparameterOptimizationFacade(scenario=scenario, target_function=evaluate_candidate, overwrite=True)
        incumbent = facade.optimize()
    best_config = {'learning_rate': float(incumbent['learning_rate']), 'weight_decay': float(incumbent['weight_decay']),
                   'loss_weight': float(incumbent['loss_weight'])}
    print('SMAC optimization completed')
    print(best_config)
    return best_config


def train_and_save_model(model_id: str, raw_data: np.ndarray, fft_data: np.ndarray, labels: np.ndarray,
                         depths: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, device: torch.device,
                         hyperparameters: Dict[str, float], output_root: Path) -> Dict[str, object]:
    set_reproducibility(SETTINGS['seed'])
    train_loader, val_loader = make_loaders(raw_data, fft_data, labels, depths, train_idx, val_idx,
                                            SETTINGS['batch_size'], SETTINGS['seed'])
    model = AblationModel(model_id).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'],
                           weight_decay=hyperparameters['weight_decay'])
    criterion = AblationLoss(hyperparameters['loss_weight'])
    best_value = -float('inf') if model_id == 'C' else float('inf')
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    print('\n' + '=' * 88)
    print(f'Training Model {model_id}')
    print('=' * 88)
    for epoch in range(1, SETTINGS['epochs'] + 1):
        model.train()
        train_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            raw = batch['raw'].to(device, non_blocking=True)
            fft_input = batch['fft'].to(device, non_blocking=True)
            batch_labels = batch['label'].to(device, non_blocking=True)
            batch_depths = batch['depth'].to(device, non_blocking=True)
            valid = batch['valid_depth'].to(device, non_blocking=True)
            outputs = model(raw, fft_input)
            loss, _, _ = criterion(outputs, batch_labels, batch_depths, valid)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batch_count += 1
        metrics, _ = evaluate_model(model, val_loader, criterion, device)
        if model_id == 'C':
            is_better = metrics['accuracy'] > best_value
            current_value = metrics['accuracy']
        else:
            is_better = metrics['mae'] < best_value
            current_value = metrics['mae']
        if is_better:
            best_value = current_value
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_state = copy.deepcopy(model.state_dict())
        acc_text = f"{metrics['accuracy'] * 100:.2f}%" if math.isfinite(metrics['accuracy']) else 'N/A'
        mae_text = f"{metrics['mae']:.4f}" if math.isfinite(metrics['mae']) else 'N/A'
        print(f"Model {model_id} | Epoch {epoch:03d}/{SETTINGS['epochs']} | "
              f"Train Loss={train_loss / max(batch_count, 1):.4f} | "
              f"Val Loss={metrics['loss']:.4f} | Acc={acc_text} | "
              f"MAE={mae_text}")
    if best_state is None:
        raise RuntimeError(f'No valid best checkpoint for Model {model_id}.')
    model.load_state_dict(best_state)
    checkpoint_path = output_root / f'Model_{model_id}_best.pth'
    torch.save({'model_id': model_id, 'epoch': best_epoch, 'model_state_dict': best_state,
                'metrics': best_metrics}, checkpoint_path)
    if model_id == 'H':
        _, arrays = evaluate_model(model, val_loader, criterion, device)
        plot_confusion_matrix(arrays['cls_true'], arrays['cls_pred'], output_root / 'MTL_TLV_confusion_matrix.png')
        plot_regression_curve(arrays['reg_true'], arrays['reg_pred'], output_root / 'MTL_TLV_regression_curve.png')
    rule = 'best accuracy' if model_id == 'C' else 'best MAE'
    print(f'Saved {checkpoint_path.name} | selection={rule} | epoch={best_epoch}')

    return {
        'model': model_id,
        'configuration': MODEL_DESCRIPTIONS[model_id],
        'accuracy': best_metrics['accuracy'],
        'mae': best_metrics['mae'],
        'selected_epoch': best_epoch,
    }


def print_ablation_summary(results: List[Dict[str, object]]) -> None:
    print('\n' + '=' * 122)
    print('Ablation summary')
    print('=' * 122)
    print(
        f"{'Model':<7}"
        f"{'Configuration':<66}"
        f"{'Recognition accuracy (%)':>29}"
        f"{'Regression MAE':>20}"
    )
    for result in results:
        accuracy = float(result['accuracy'])
        mae = float(result['mae'])
        accuracy_text = (
            f"{accuracy * 100:.2f}"
            if math.isfinite(accuracy)
            else '—'
        )
        mae_text = (
            f"{mae:.2f}"
            if math.isfinite(mae)
            else '—'
        )
        print(
            f"{str(result['model']):<7}"
            f"{str(result['configuration']):<66}"
            f"{accuracy_text:>29}"
            f"{mae_text:>20}"
        )
    print('=' * 122)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / SETTINGS['data_dir']).resolve()
    output_root = (script_dir / SETTINGS['output_dir']).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required but is not available.')
    device = torch.device(SETTINGS['device'])
    set_reproducibility(SETTINGS['seed'])
    raw_data, clean_labels, depths, groups = read_data(data_dir)
    labels = inject_label_noise(clean_labels, SETTINGS['label_noise_rate'], SETTINGS['seed'])
    train_idx, val_idx = make_split(labels, groups, SETTINGS['val_ratio'], SETTINGS['split_seed'])
    fft_data = build_fft_features(raw_data, SETTINGS['n_fft'], SETTINGS['fft_bins'])
    hyperparameters = run_smac_search(raw_data, fft_data, labels, depths, train_idx, val_idx, device)
    results: List[Dict[str, object]] = []
    for model_id in SETTINGS['models']:
        result = train_and_save_model(
            model_id,
            raw_data,
            fft_data,
            labels,
            depths,
            train_idx,
            val_idx,
            device,
            hyperparameters,
            output_root,
        )
        results.append(result)

    print_ablation_summary(results)

    print('\nSaved files:')
    for path in sorted(output_root.iterdir()):
        if path.is_file():
            print(f'  {path.name}')


if __name__ == '__main__':

    main()