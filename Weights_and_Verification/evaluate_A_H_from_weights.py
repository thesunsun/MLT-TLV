"""
Independent checkpoint verification for MTL-TLV Models A-H.

This script:
1. reconstructs the same preprocessing and validation split as training;
2. loads Model_A_best.pth ... Model_H_best.pth;
3. recomputes Accuracy, Precision, Recall, and MAE from model inference;
4. compares recomputed metrics with those stored in each checkpoint.

No retraining is performed.
"""

from __future__ import annotations
import math
import os
import platform
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import fft
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
MODEL_DESCRIPTIONS: Dict[str, str] = {'A': 'Temporal single-stream baseline (classification + regression)', 'B': 'Spectral single-stream baseline (classification + regression)', 'C': 'Dual-stream naive concatenation, classification only', 'D': 'Dual-stream naive concatenation, regression only', 'E': 'Dual-stream naive multi-task integration', 'F': 'Bidirectional fusion only, no TAFR', 'G': 'TAFR only, no bidirectional fusion', 'H': 'Proposed MTL-TLV (bidirectional fusion + TAFR)'}
VALID_MODELS = tuple(MODEL_DESCRIPTIONS)
SETTINGS = {'data_dir': '.', 'weights_dir': 'MTL-TLV-With-Model-A-G-Output', 'models': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'batch_size': 512, 'seed': 35, 'split_seed': 35, 'val_ratio': 0.1, 'label_noise_rate': 0.012, 'fft_bins': 50, 'n_fft': 512, 'device': 'cuda:0' if torch.cuda.is_available() else 'cpu', 'comparison_tolerance': 1e-06}

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
        return {'raw': self.raw_data[idx], 'fft': self.fft_data[idx], 'label': self.labels[idx], 'depth': self.depths[idx], 'valid_depth': self.valid_depths[idx]}

def set_reproducibility(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    """
    Read the six CSV files using exactly the same construction as training.
    """
    filenames = {'normal_1': 'normal_data_1.csv', 'normal_2': 'normal_data_2.csv', 'void_1': 'void_data_1.csv', 'void_2': 'void_data_2.csv', 'void_depth_1': 'void_depth_1.csv', 'void_depth_2': 'void_depth_2.csv'}
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
    groups = np.concatenate([np.zeros(normal_1.shape[1], dtype=np.int64), np.ones(normal_2.shape[1], dtype=np.int64), np.zeros(void_1.shape[1], dtype=np.int64), np.ones(void_2.shape[1], dtype=np.int64)])
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
    """
    Reproduce the fixed domain split used by the training script.
    """
    group0 = np.where(groups == 0)[0]
    group1 = np.where(groups == 1)[0]
    val_size = int(len(labels) * val_ratio)
    if val_size <= 0 or val_size >= len(group0):
        raise ValueError('Invalid validation size for the fixed domain split.')
    train_group0, val_idx = train_test_split(group0, test_size=val_size, random_state=split_seed, stratify=labels[group0])
    train_idx = np.concatenate([group1, train_group0])
    return (np.sort(train_idx), np.sort(val_idx))

def make_validation_loader(raw_data: np.ndarray, fft_data: np.ndarray, labels: np.ndarray, depths: np.ndarray, val_idx: np.ndarray, batch_size: int) -> DataLoader:
    val_set = GPRDataset(raw_data[val_idx], fft_data[val_idx], labels[val_idx], depths[val_idx])
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)

class ResNet1D(nn.Module):
    """1D ResNet-18-style feature extractor returning a 512-D vector."""

    def __init__(self, block: type[BasicBlock1D]=BasicBlock1D, layers: Sequence[int]=(2, 2, 2, 2), feature_dim: int=512) -> None:
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
    """Time-frequency dual-domain fusion operation."""

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

    def __init__(self, channel: int, reduction: int=16) -> None:
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.routing_gate = nn.Sequential(nn.Linear(channel, hidden, bias=False), nn.ReLU(inplace=True), nn.Linear(hidden, channel, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights = self.routing_gate(x)
        return x + x * routing_weights

class ClassificationHead(nn.Module):

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RegressionHead(nn.Module):

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 1))

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

def evaluate_checkpoint(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Recompute the reported metrics directly from predictions.

    No metrics stored in the checkpoint are used here.
    """
    model.eval()
    cls_true: List[int] = []
    cls_pred: List[int] = []
    reg_true: List[float] = []
    reg_pred: List[float] = []
    with torch.no_grad():
        for batch in loader:
            raw = batch['raw'].to(device, non_blocking=True)
            fft_input = batch['fft'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            depths = batch['depth'].to(device, non_blocking=True)
            valid = batch['valid_depth'].to(device, non_blocking=True)
            outputs = model(raw, fft_input)
            if outputs['cls'] is not None:
                prediction = outputs['cls'].argmax(dim=1)
                cls_true.extend(labels.cpu().tolist())
                cls_pred.extend(prediction.cpu().tolist())
            if outputs['reg'] is not None and bool(valid.any()):
                reg_true.extend(depths[valid].cpu().numpy().tolist())
                reg_pred.extend(outputs['reg'][valid].squeeze(1).cpu().numpy().tolist())
    accuracy = float(accuracy_score(cls_true, cls_pred)) if cls_true else float('nan')
    precision = float(precision_score(cls_true, cls_pred, zero_division=0)) if cls_true else float('nan')
    recall = float(recall_score(cls_true, cls_pred, zero_division=0)) if cls_true else float('nan')
    mae = float(mean_absolute_error(reg_true, reg_pred)) if reg_true else float('nan')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'mae': mae}

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    """
    Load checkpoint safely across multiple PyTorch versions.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = {'model_id', 'epoch', 'model_state_dict', 'metrics'}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise KeyError(f'{checkpoint_path.name} is missing checkpoint keys: {sorted(missing)}')
    return checkpoint

def values_match(reproduced: float, stored: float, tolerance: float) -> bool:
    """
    Compare two metric values while correctly handling NaN.
    """
    reproduced = float(reproduced)
    stored = float(stored)
    if math.isnan(reproduced) and math.isnan(stored):
        return True
    if math.isnan(reproduced) or math.isnan(stored):
        return False
    return abs(reproduced - stored) <= tolerance

def format_percent(value: float) -> str:
    return f'{value * 100:.2f}' if math.isfinite(value) else '—'

def format_mae(value: float) -> str:
    return f'{value:.2f}' if math.isfinite(value) else '—'

def print_environment(device: torch.device) -> None:
    print('=' * 110)
    print('MTL-TLV A-H CHECKPOINT REPRODUCTION')
    print('=' * 110)
    print(f'Operating system : {platform.platform()}')
    print(f'Python           : {platform.python_version()}')
    print(f'PyTorch          : {torch.__version__}')
    print(f'CUDA runtime     : {torch.version.cuda}')
    print(f'cuDNN            : {torch.backends.cudnn.version()}')
    print(f'Device           : {device}')
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f'GPU              : {props.name}')
        print(f'GPU memory       : {props.total_memory / 1024 ** 3:.2f} GiB')

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / SETTINGS['data_dir']).resolve()
    weights_dir = (script_dir / SETTINGS['weights_dir']).resolve()
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights directory not found:\n  {weights_dir}\n\nPlease place Model_A_best.pth ... Model_H_best.pth in this directory, or change SETTINGS['weights_dir'].")
    device = torch.device(SETTINGS['device'])
    set_reproducibility(SETTINGS['seed'])
    print_environment(device)
    raw_data, clean_labels, depths, groups = read_data(data_dir)
    labels = inject_label_noise(clean_labels, SETTINGS['label_noise_rate'], SETTINGS['seed'])
    train_idx, val_idx = make_split(labels, groups, SETTINGS['val_ratio'], SETTINGS['split_seed'])
    fft_data = build_fft_features(raw_data, SETTINGS['n_fft'], SETTINGS['fft_bins'])
    val_loader = make_validation_loader(raw_data, fft_data, labels, depths, val_idx, SETTINGS['batch_size'])
    results: List[Dict[str, object]] = []
    all_passed = True
    for model_id in SETTINGS['models']:
        checkpoint_path = weights_dir / f'Model_{model_id}_best.pth'
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')
        checkpoint = load_checkpoint(checkpoint_path, device)
        checkpoint_model_id = str(checkpoint['model_id']).upper()
        if checkpoint_model_id != model_id:
            raise ValueError(f'{checkpoint_path.name}: checkpoint model_id={checkpoint_model_id}, expected {model_id}.')
        model = AblationModel(model_id).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        reproduced = evaluate_checkpoint(model, val_loader, device)
        stored = checkpoint['metrics']
        tolerance = float(SETTINGS['comparison_tolerance'])
        checks = {metric: values_match(reproduced[metric], float(stored[metric]), tolerance) for metric in ('accuracy', 'precision', 'recall', 'mae')}
        checkpoint_passed = all(checks.values())
        all_passed = all_passed and checkpoint_passed
        print('\n' + '=' * 110)
        print(f'Model {model_id}: {MODEL_DESCRIPTIONS[model_id]}')
        print('=' * 110)
        print(f'Checkpoint       : {checkpoint_path.name}')
        print('-' * 110)
        print(f"{'Metric':<14}{'Recomputed from weights':>28}{'Stored in checkpoint':>26}{'Consistency':>18}")
        print('-' * 110)
        for metric in ('accuracy', 'precision', 'recall', 'mae'):
            rep = float(reproduced[metric])
            ref = float(stored[metric])
            if metric in ('accuracy', 'precision', 'recall'):
                rep_text = f'{rep * 100:.8f}%' if math.isfinite(rep) else 'N/A'
                ref_text = f'{ref * 100:.8f}%' if math.isfinite(ref) else 'N/A'
            else:
                rep_text = f'{rep:.8f}' if math.isfinite(rep) else 'N/A'
                ref_text = f'{ref:.8f}' if math.isfinite(ref) else 'N/A'
            status = 'PASS' if checks[metric] else 'FAIL'
            print(f'{metric:<14}{rep_text:>28}{ref_text:>26}{status:>18}')
        print('-' * 110)
        print(f"Checkpoint verification: {('PASS' if checkpoint_passed else 'FAIL')}")
        results.append({'model': model_id, 'configuration': MODEL_DESCRIPTIONS[model_id], 'selected_epoch': int(checkpoint['epoch']), 'accuracy': reproduced['accuracy'], 'precision': reproduced['precision'], 'recall': reproduced['recall'], 'mae': reproduced['mae'], 'verification': 'PASS' if checkpoint_passed else 'FAIL'})
        del model, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print('\n\n' + '=' * 136)
    print('TABLE 5 REPRODUCTION DIRECTLY FROM THE EIGHT RELEASED CHECKPOINTS')
    print('=' * 136)
    print(f"{'Model':<7}{'Configuration':<66}{'Accuracy (%)':>15}{'Precision (%)':>16}{'Recall (%)':>14}{'MAE':>10}{'Check':>8}")
    print('-' * 136)
    for result in results:
        accuracy = float(result['accuracy'])
        precision = float(result['precision'])
        recall = float(result['recall'])
        mae = float(result['mae'])
        print(f"{str(result['model']):<7}{str(result['configuration']):<66}{format_percent(accuracy):>15}{format_percent(precision):>16}{format_percent(recall):>14}{format_mae(mae):>10}{str(result['verification']):>8}")
    print('=' * 136)
    print('\n' + '=' * 110)
    if all_passed:
        print('OVERALL CHECK: PASS\nAll eight checkpoints reproduce the metrics stored at training time within the specified tolerance.')
    else:
        print('OVERALL CHECK: FAIL\nAt least one recomputed metric differs from the corresponding metric stored in its checkpoint.')
    print('=' * 110)
if __name__ == '__main__':
    main()
