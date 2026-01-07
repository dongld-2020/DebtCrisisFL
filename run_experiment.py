import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import csv
import os
from collections import Counter

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File path
file_path = 'data/raw_data.xlsx'

# Load data with remapping class 2,3 to 1
xl = pd.ExcelFile(file_path)
sheet_names = xl.sheet_names

# Standardize features
scaler = StandardScaler()

# Global test set with remapping
global_test_df = pd.read_excel(xl, sheet_names[0])
y_test_original = global_test_df.iloc[:, -1].values.astype(np.int64)
y_test = np.where((y_test_original == 2) | (y_test_original == 3), 1, y_test_original)
y_test = np.clip(y_test, 0, 1)
X_test = scaler.fit_transform(global_test_df.iloc[:, :-1].values.astype(np.float32))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Test set after remapping: Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)} (total {len(y_test)})")

# Load client data with remapping
client_data = []
client_class_dist = []
for sheet in sheet_names[1:]:
    df = pd.read_excel(xl, sheet)
    y_original = df.iloc[:, -1].values.astype(np.int64)
    y = np.where((y_original == 2) | (y_original == 3), 1, y_original)
    y = np.clip(y, 0, 1)
    X = scaler.transform(df.iloc[:, :-1].values.astype(np.float32))
    class_counts = Counter(y)
    client_class_dist.append(class_counts)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
    client_data.append((loader, len(dataset)))

num_clients = len(client_data)
print(f"Number of clients: {num_clients}")


# DeepNN model
input_dim = X_test.shape[1]
hidden_dim1, hidden_dim2, hidden_dim3 = 256, 128, 64
output_dim = 2

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
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
        x = self.fc4(x)
        return x
        
# TransformerNN model
class TransformerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.5):
        super(TransformerNN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Add singleton sequence dimension: (batch_size, input_dim) -> (batch_size, 1, hidden_dim)
        x = self.embedding(x)
        x = self.relu(x)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, hidden_dim)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.squeeze(1)  # Remove sequence dimension: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)  # Shape: (batch_size, output_dim)
        return x

# CNN1D model
class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Residual Neural Network
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.skip_connection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        identity = x
        if self.skip_connection:
            identity = self.skip_connection(identity)
        
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        out = self.relu(out)
        return out

class ResidualNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, output_dim=2):
        super(ResidualNN, self).__init__()
        self.block1 = ResidualBlock(input_dim, hidden_dim1)
        self.block2 = ResidualBlock(hidden_dim1, hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

# New model: Simple Feedforward Neural Network (FNN)
class SimpleFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, output_dim=2):
        super(SimpleFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Enhanced Focal Loss with adaptive gamma
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma_min=1.0, gamma_max=3.0):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.alpha = alpha
        self.step = 0
    
    def forward(self, inputs, targets):
        progress = min(1.0, self.step / 1000)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * progress
        self.step += 1
        
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

# Class weights
def compute_class_weights(y, scale_factor=3.0):
    unique, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(unique) * scale_factor
    return torch.tensor(weights, dtype=torch.float32)

class_weights = compute_class_weights(y_test)
focal_criterion = AdaptiveFocalLoss(alpha=class_weights, gamma_min=1.0, gamma_max=5.0)

# Enhanced client quality scoring
def client_quality_score_enhanced(class_dist, min_samples=10):
    if 1 not in class_dist or 0 not in class_dist:
        return 0.1
    
    minority_class = 1 if class_dist[1] < class_dist[0] else 0
    minority_count = class_dist[minority_class]
    majority_count = class_dist[1 - minority_class]
    
    if minority_count < min_samples:
        return 0.1
    
    balance_ratio = minority_count / (majority_count + 1e-8)
    balance_score = min(1.0, balance_ratio * 2)
    
    minority_reward = min(1.0, minority_count / 50)
    
    total_score = balance_score * 0.6 + minority_reward * 0.4
    return total_score

# Dynamic learning rate
def get_client_lr(client_quality, base_lr=0.0005):
    return base_lr * (0.5 + client_quality * 1.5)

# Create balanced test loader
def create_balanced_test_loader(X_test, y_test, samples_per_class=500):
    class_0_idx = np.where(y_test == 0)[0]
    class_1_idx = np.where(y_test == 1)[0]
    
    sampled_0 = np.random.choice(class_0_idx, min(samples_per_class, len(class_0_idx)), replace=False)
    sampled_1 = np.random.choice(class_1_idx, min(samples_per_class, len(class_1_idx)), replace=False)
    
    balanced_idx = np.concatenate([sampled_0, sampled_1])
    X_balanced = X_test[balanced_idx]
    y_balanced = y_test[balanced_idx]
    
    return DataLoader(TensorDataset(torch.tensor(X_balanced), torch.tensor(y_balanced)), 
                      batch_size=64, shuffle=False)

balanced_test_loader = create_balanced_test_loader(X_test, y_test)

# FedProx local training function
def train_local_fedprox(model, global_model, loader, client_quality=0.5, epochs=5, proximal_mu=0.1):
    lr = get_client_lr(client_quality)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = 1
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.95 ** (epoch - warmup_epochs)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    global_model_state = global_model.state_dict()
    model.train()
    total_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_batches = 0
        for data, target in loader:
            if data.size(0) < 2:
                continue
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            loss_standard = focal_criterion(output, target.long())
            proximal_term = 0
            for name, param in model.named_parameters():
                proximal_term += torch.sum(torch.pow(param - global_model_state[name], 2))
            
            loss = loss_standard + (proximal_mu / 2) * proximal_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        if epoch_batches > 0:
            total_loss += epoch_loss / epoch_batches
            num_batches += 1
        
        scheduler.step()
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

# FedNoLoWe local training function
def train_local_model_fednolowe(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if data.size(0) < 2:
                continue
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
            epoch_samples += data.size(0)
        total_loss += epoch_loss
        total_samples += epoch_samples
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return model, avg_loss

# FedNoLoWe aggregation function
def aggregate_nolowe(global_model, client_models, feedback_train_losses):
    print("Clients train loss: ", feedback_train_losses)
    
    feedback_train_losses = np.array(feedback_train_losses)
    if feedback_train_losses.sum() == 0:
        print("Warning: clients_train_loss sum to 0. Assigning equal weights.")
        feedback_train_losses = np.ones(len(feedback_train_losses)) / len(feedback_train_losses)
    else:
        feedback_train_losses /= feedback_train_losses.sum()
        feedback_train_losses = 1 - feedback_train_losses
        feedback_train_losses /= feedback_train_losses.sum()
    
    print("Normalized clients train loss: ", feedback_train_losses)
    
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) 
                    for key in global_model.state_dict().keys()}
    
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            weighted_sum[key] += client_models[i].state_dict()[key].float() * feedback_train_losses[i]
        
    global_model.load_state_dict(weighted_sum)
    return global_model

# Evaluate function
def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, num_batches = 0, 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.long())
            total_loss += loss.item()
            num_batches += 1
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    report_str = classification_report(all_targets, all_preds, labels=[0,1], zero_division=0)
    
    return avg_loss, f1, cm, accuracy, balanced_acc, report_str

# Save predictions for analysis
def save_predictions(model, loader, filename='predictions_analysis.csv'):
    model.eval()
    all_preds, all_probs, all_targets, all_indices = [], [], [], []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_indices.extend(range(batch_idx * loader.batch_size, batch_idx * loader.batch_size + len(data)))
    
    results_df = pd.DataFrame({
        'index': all_indices,
        'true_label': all_targets,
        'predicted_label': all_preds,
        'prob_class_0': [p[0] for p in all_probs],
        'prob_class_1': [p[1] for p in all_probs]
    })
    
    results_df.to_csv(filename, index=False)
    print(f"📁 Predictions saved to {filename}")

# FL setup
num_rounds = 50
fraction_min, fraction_max = 0.3, 0.4
best_f1, best_balanced_acc = 0.0, 0.0
best_model_path = 'best_global_model_enhanced.pth'
local_epochs=5
# User input for model selection
while True:
    model_choice = input("Choose model (DeepNN, TransformerNN, CNN1D, ResidualNN, or SimpleFNN): ").strip().lower()
    if model_choice in ['deepnn', 'transformernn', 'cnn1d', 'residualnn', 'simplefnn']:
        break
    print("Invalid choice. Please enter 'DeepNN', 'TransformerNN', 'CNN1D', 'ResidualNN', or 'SimpleFNN'.")

# Initialize global model based on user choice
if model_choice == 'deepnn':
    global_model = DeepNN().to(device)
elif model_choice == 'transformernn':
    global_model = TransformerNN(input_dim=input_dim).to(device)
elif model_choice == 'cnn1d':
    global_model = CNN1D(input_dim=input_dim).to(device)
elif model_choice == 'residualnn':
    global_model = ResidualNN(input_dim=input_dim).to(device)
else: # model_choice == 'simplefnn'
    global_model = SimpleFNN(input_dim=input_dim).to(device)


# Filter clients and calculate quality scores
eligible_clients = [i for i in range(num_clients) if client_class_dist[i].get(1, 0) >= 8]
quality_scores = [client_quality_score_enhanced(client_class_dist[i]) for i in eligible_clients]
if sum(quality_scores) > 0:
    probabilities = [score / sum(quality_scores) for score in quality_scores]
else:
    probabilities = [1.0 / len(eligible_clients)] * len(eligible_clients)

print(f"Eligible clients: {len(eligible_clients)}")
print(f"Quality score range: min={min(quality_scores):.4f}, max={max(quality_scores):.4f}")

# User input for algorithm selection
while True:
    algo_choice = input("Choose federated learning algorithm (FedProx or FedNoLoWe): ").strip().lower()
    if algo_choice in ['fedprox', 'fednolowe']:
        break
    print("Invalid choice. Please enter 'FedProx' or 'FedNoLoWe'.")


csv_file = f"fl_metrics_{model_choice.upper()}_{algo_choice.upper()}_epochs_{local_epochs}_frac_{fraction_min}_{fraction_max}_rounds_{num_rounds}.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Round', 'Avg_Train_Loss', 'Val_Loss', 'F1_Score', 'Accuracy', 'Balanced_Acc', 
                     'Confusion_Matrix', 'Balanced_F1', 'Balanced_Acc', 'Classification_Report', 'Model'])
                     
# Training loop
for round_num in range(1, num_rounds + 1):
    num_selected = max(1, int(random.uniform(fraction_min, fraction_max) * len(eligible_clients)))
    selected_clients = np.random.choice(eligible_clients, size=min(num_selected, len(eligible_clients)), 
                                        p=probabilities, replace=False)
    
    print(f"\nRound {round_num}: Selected {len(selected_clients)} clients")
    
    client_models = []
    train_losses = []
    
    for client_idx in selected_clients:
        if model_choice == 'deepnn':
            local_model = DeepNN().to(device)
        elif model_choice == 'transformernn':
            local_model = TransformerNN(input_dim=input_dim).to(device)
        elif model_choice == 'cnn1d':
            local_model = CNN1D(input_dim=input_dim).to(device)
        elif model_choice == 'residualnn':
            local_model = ResidualNN(input_dim=input_dim).to(device)
        else: # model_choice == 'simplefnn'
            local_model = SimpleFNN(input_dim=input_dim).to(device)
            
        local_model.load_state_dict(global_model.state_dict())
        loader, data_size = client_data[client_idx]
        
        if data_size < 64:
            continue
            
        client_quality = quality_scores[eligible_clients.index(client_idx)]
        
        if algo_choice == 'fedprox':
            loss = train_local_fedprox(local_model, global_model, loader, client_quality, epochs=local_epochs, proximal_mu=0.5)
            if loss == float('inf') or loss < 1e-4:
                continue
            client_models.append((local_model.state_dict(), data_size, client_quality))
        else:  # FedNoLoWe
            optimizer = optim.Adam(local_model.parameters(), lr=get_client_lr(client_quality), weight_decay=1e-4)
            local_model, loss = train_local_model_fednolowe(local_model, loader, focal_criterion, optimizer, epochs=local_epochs)
            if loss == float('inf') or loss < 1e-4:
                continue
            client_models.append(local_model)
        
        print(f"  Client {sheet_names[client_idx+1]} (quality: {client_quality:.3f}) training loss: {loss:.4f}")
        train_losses.append(loss)
    
    if not client_models:
        print("No valid clients this round.")
        continue
    
    # Aggregation
    if algo_choice == 'fedprox':
        total_weighted_size = sum(size * quality for _, size, quality in client_models)
        avg_state_dict = {}
        for key in global_model.state_dict().keys():
            weighted_sum = sum(sd[key] * size * quality for sd, size, quality in client_models)
            avg_state_dict[key] = weighted_sum / total_weighted_size
        global_model.load_state_dict(avg_state_dict)
    else:  # FedNoLoWe
        global_model = aggregate_nolowe(global_model, client_models, train_losses)
    
    # Evaluate on both original and balanced test sets
    val_loss, f1, cm, accuracy, balanced_acc, report_str = evaluate(global_model, test_loader)
    _, balanced_f1, _, _, balanced_balanced_acc, _ = evaluate(global_model, balanced_test_loader)
    cm_str = str(cm.tolist())
    
    print(f"Round {round_num} metrics:")
    print(f"Avg Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}")
    print(f"F1: {f1:.4f}, Acc: {accuracy:.4f}, Balanced Acc: {balanced_acc:.4f}")
    print(f"Balanced F1: {balanced_f1:.4f}, Balanced Balanced Acc: {balanced_balanced_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, np.mean(train_losses), val_loss, f1, accuracy, balanced_acc, 
                         cm_str, balanced_f1, balanced_balanced_acc, report_str, model_choice.upper()])
    
    # Save best model without early stopping
    if balanced_f1 > best_f1:
        best_f1 = balanced_f1
        best_balanced_acc = balanced_balanced_acc
        torch.save(global_model.state_dict(), best_model_path)
        print(f"🎯 New best Balanced F1: {balanced_f1:.4f}, Balanced Acc: {balanced_balanced_acc:.4f}")
    else:
        print(f"⏳ No improvement in this round.")

print(f"\n✅ FL completed after {num_rounds} rounds using {algo_choice.upper()} with {model_choice.upper()}.")

# Final evaluation with the best model found
if os.path.exists(best_model_path):
    global_model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\n📊 Final evaluation with best model.")

final_val_loss, final_f1, final_cm, final_accuracy, final_balanced_acc, final_report = evaluate(global_model, test_loader)
final_balanced_loss, final_balanced_f1, _, _, final_balanced_balanced_acc, _ = evaluate(global_model, balanced_test_loader)

print(f"\n🎉 Final Results with Best Model:")
print(f"Test Loss: {final_val_loss:.4f}, F1: {final_f1:.4f}, Acc: {final_accuracy:.4f}, Balanced Acc: {final_balanced_acc:.4f}")
print(f"Balanced Test Loss: {final_balanced_loss:.4f}, Balanced F1: {final_balanced_f1:.4f}, Balanced Balanced Acc: {final_balanced_balanced_acc:.4f}")
print(f"Confusion Matrix:\n{final_cm}")
print(f"Classification Report:\n{final_report}")

print(f"\n✅ FL completed. Best Balanced F1: {best_f1:.4f}, Best Balanced Acc: {best_balanced_acc:.4f}")

save_predictions(global_model, test_loader)