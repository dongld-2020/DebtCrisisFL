import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
file_path = 'fl_data_exported_14092025.xlsx'

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
        x = self.embedding(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.fc(x)
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

# Simple Feedforward Neural Network
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

# Initialize global model based on user choice
def initialize_global_model(model_choice, input_dim):
    if model_choice == 'deepnn':
        return DeepNN().to(device)
    elif model_choice == 'transformernn':
        return TransformerNN(input_dim=input_dim).to(device)
    elif model_choice == 'cnn1d':
        return CNN1D(input_dim=input_dim).to(device)
    elif model_choice == 'residualnn':
        return ResidualNN(input_dim=input_dim).to(device)
    else:  # model_choice == 'simplefnn'
        return SimpleFNN(input_dim=input_dim).to(device)

# User input for model selection
while True:
    model_choice = input("Choose model (DeepNN, TransformerNN, CNN1D, ResidualNN, or SimpleFNN): ").strip().lower()
    if model_choice in ['deepnn', 'transformernn', 'cnn1d', 'residualnn', 'simplefnn']:
        break
    print("Invalid choice. Please enter 'DeepNN', 'TransformerNN', 'CNN1D', 'ResidualNN', or 'SimpleFNN'.")

# User input for model path
while True:
    model_path = input("Enter the path to the .pth model file: ").strip()
    if os.path.exists(model_path):
        break
    print(f"File {model_path} not found. Please enter a valid path.")

# Initialize global model
global_model = initialize_global_model(model_choice, input_dim)

# Load the pre-trained model
try:
    global_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Successfully loaded model from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Final evaluation
print("\n🎉 Performing Final Evaluation")
# Evaluate on full test set
final_val_loss, final_f1, final_cm, final_accuracy, final_balanced_acc, final_report = evaluate(global_model, test_loader)
# Evaluate on balanced test set
final_balanced_loss, final_balanced_f1, _, _, final_balanced_balanced_acc, _ = evaluate(global_model, balanced_test_loader)

print(f"\nFinal Results for Loaded Model:")
print(f"Test Loss: {final_val_loss:.4f}, F1: {final_f1:.4f}, Accuracy: {final_accuracy:.4f}, Balanced Accuracy: {final_balanced_acc:.4f}")
print(f"Balanced Test Loss: {final_balanced_loss:.4f}, Balanced F1: {final_balanced_f1:.4f}, Balanced Balanced Accuracy: {final_balanced_balanced_acc:.4f}")
print(f"Confusion Matrix:\n{final_cm}")
print(f"Classification Report:\n{final_report}")

# Save predictions
predictions_file = f"predictions_final_{model_choice.upper()}.csv"
save_predictions(global_model, test_loader, predictions_file)
print(f"✅ Final evaluation completed. Predictions saved to {predictions_file}")