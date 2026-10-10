"""
fl_common.py
================
Các thành phần dùng chung cho run_grid_search.py và run_baselines.py:
kiến trúc mô hình (giữ nguyên như bản gốc để so sánh công bằng), Adaptive
Focal Loss, tính class weight theo kiểu "federated-safe" (chỉ dùng số đếm
nhãn tổng hợp từ các client TRAIN, KHÔNG dùng nhãn test), các hàm huấn
luyện FedProx / FedNoLoWe, và hàm evaluate() nhận vào loader bất kỳ (Val
hoặc Test) để tách bạch rõ vai trò hai tập này.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, confusion_matrix, accuracy_score,
                              classification_report, balanced_accuracy_score)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- Mô hình --------------------------------
class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)


class TransformerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.5, output_dim=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="relu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.embedding(x))
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        identity = self.skip(x) if self.skip else x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)


class ResidualNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, output_dim=2):
        super().__init__()
        self.block1 = ResidualBlock(input_dim, hidden_dim1)
        self.block2 = ResidualBlock(hidden_dim1, hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.block1(x))
        x = self.dropout(self.block2(x))
        return self.fc_out(x)


def build_model(name, input_dim):
    name = name.lower()
    if name == "deepnn":
        return DeepNN(input_dim).to(device)
    if name == "transformernn":
        return TransformerNN(input_dim).to(device)
    if name == "residualnn":
        return ResidualNN(input_dim).to(device)
    raise ValueError(f"Kiến trúc không hỗ trợ: {name}")


# ----------------------------- Loss ------------------------------------
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma_min=1.0, gamma_max=3.0):
        super().__init__()
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.alpha = alpha.to(device)
        self.step = 0

    def forward(self, inputs, targets):
        progress = min(1.0, self.step / 1000)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * progress
        self.step += 1
        ce = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")(inputs, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()


def compute_class_weights_federated_safe(client_train_counts_df, scale_factor=3.0):
    """Tính class weight KHÔNG dùng nhãn test.

    client_train_counts_df: DataFrame có cột n_class0, n_class1 (đã được
    tạo bởi prepare_debtcrisisfl.py -> client_train_class_counts.csv).
    Đây là tổng hợp scalar từ các client TRAIN -- tương thích với giả định
    FL (server chỉ nhận số đếm, không nhận dữ liệu thô hay nhãn test).
    """
    n0 = int(client_train_counts_df["n_class0"].sum())
    n1 = int(client_train_counts_df["n_class1"].sum())
    counts = np.array([n0, n1], dtype=np.float64)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts) * scale_factor
    return torch.tensor(weights, dtype=torch.float32), n0, n1


# ------------------------- Client quality score -------------------------
def client_quality_score_enhanced(class_dist, min_samples=10):
    if 1 not in class_dist or 0 not in class_dist:
        return 0.1
    minority_class = 1 if class_dist[1] < class_dist[0] else 0
    minority_count = class_dist[minority_class]
    majority_count = class_dist[1 - minority_class]
    if minority_count < min_samples:
        return 0.1
    balance_score = min(1.0, (minority_count / (majority_count + 1e-8)) * 2)
    minority_reward = min(1.0, minority_count / 50)
    return balance_score * 0.6 + minority_reward * 0.4


def get_client_lr(client_quality, base_lr=0.0005):
    return base_lr * (0.5 + client_quality * 1.5)


# ----------------------------- Training ----------------------------------
def train_local_fedprox(model, global_model, loader, criterion, client_quality=0.5,
                         epochs=5, proximal_mu=0.1):
    lr = get_client_lr(client_quality)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) if e < 1 else 0.95 ** (e - 1))
    global_state = global_model.state_dict()
    model.train()
    total_loss, num_batches = 0.0, 0
    for epoch in range(epochs):
        epoch_loss, epoch_batches = 0.0, 0
        for data, target in loader:
            if data.size(0) < 2:
                continue
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            prox = sum(torch.sum((p - global_state[n]) ** 2) for n, p in model.named_parameters())
            loss = loss + (proximal_mu / 2) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_batches += 1
        if epoch_batches > 0:
            total_loss += epoch_loss / epoch_batches
            num_batches += 1
        scheduler.step()
    return total_loss / num_batches if num_batches > 0 else float("inf")


def train_local_fednolowe(model, loader, criterion, client_quality=0.5, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=get_client_lr(client_quality), weight_decay=1e-4)
    model.train()
    total_loss, total_n = 0.0, 0
    for epoch in range(epochs):
        for data, target in loader:
            if data.size(0) < 2:
                continue
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_n += data.size(0)
    return model, (total_loss / total_n if total_n > 0 else float("inf"))


def aggregate_fedprox(global_model, client_models):
    total_w = sum(size * quality for _, size, quality in client_models)
    avg_state = {}
    for key in global_model.state_dict().keys():
        avg_state[key] = sum(sd[key] * size * quality for sd, size, quality in client_models) / total_w
    global_model.load_state_dict(avg_state)
    return global_model


def aggregate_nolowe(global_model, client_models, train_losses):
    losses = np.array(train_losses, dtype=np.float64)
    if losses.sum() == 0:
        w = np.ones(len(losses)) / len(losses)
    else:
        w = losses / losses.sum()
        w = 1 - w
        w = w / w.sum()
    weighted_sum = {k: torch.zeros_like(v, dtype=torch.float) for k, v in global_model.state_dict().items()}
    for key in weighted_sum:
        for i, m in enumerate(client_models):
            weighted_sum[key] += m.state_dict()[key].float() * w[i]
    global_model.load_state_dict(weighted_sum)
    return global_model


# ----------------------------- Evaluation ---------------------------------
def evaluate(model, loader):
    """Đánh giá trên MỘT loader duy nhất (Val hoặc Test) -- người gọi phải
    tự chỉ rõ đang đánh giá tập nào, tránh nhầm lẫn giữa chọn mô hình và
    báo cáo kết quả cuối như trong bản gốc."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, num_batches = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target.long()).item()
            num_batches += 1
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, labels=[0, 1], zero_division=0)
    return {"loss": avg_loss, "macro_f1": f1, "cm": cm, "accuracy": acc,
            "balanced_accuracy": bacc, "report": report}


def make_balanced_loader(X, y, samples_per_class=500, seed=42, batch_size=64):
    """Chỉ dùng cho BÁO CÁO PHỤ (secondary reporting), KHÔNG dùng để chọn
    hyperparameter/model -- tránh optimistic bias mà Editor/R4 đã chỉ ra."""
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    s0 = rng.choice(idx0, min(samples_per_class, len(idx0)), replace=False)
    s1 = rng.choice(idx1, min(samples_per_class, len(idx1)), replace=False)
    idx = np.concatenate([s0, s1])
    return DataLoader(TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx])),
                       batch_size=batch_size, shuffle=False)
