import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

def compute_class_weights(y, scale_factor=3.0):
    unique, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(unique) * scale_factor
    return torch.tensor(weights, dtype=torch.float32)

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

def get_client_lr(client_quality, base_lr=0.0005):
    return base_lr * (0.5 + client_quality * 1.5)

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

def evaluate(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
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

def save_predictions(model, loader, filename, device):
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