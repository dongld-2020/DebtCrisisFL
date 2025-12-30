import torch.optim as optim
import random
import numpy as np
import torch
import pandas as pd
import csv
import os
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import config
import utils
from model import DeepNN, TransformerNN, AdaptiveFocalLoss
from client import train_local_fedprox, train_local_model_fednolowe

def load_data():
    xl = pd.ExcelFile(config.file_path)
    sheet_names = xl.sheet_names
    
    # Standardize features
    scaler = utils.StandardScaler()
    
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
    
    config.input_dim = X_test.shape[1]
    return client_data, client_class_dist, test_loader, sheet_names

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

def main():
    # Set random seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    # Load data
    client_data, client_class_dist, test_loader, sheet_names = load_data()
    num_clients = len(client_data)
    print(f"Number of clients: {num_clients}")
    
    # Create balanced test loader
    X_test_np = test_loader.dataset.tensors[0].numpy()
    y_test_np = test_loader.dataset.tensors[1].numpy()
    balanced_test_loader = utils.create_balanced_test_loader(X_test_np, y_test_np)
    
    # User input for model selection
    while True:
        model_choice = input("Choose model (DeepNN or TransformerNN): ").strip().lower()
        if model_choice in ['deepnn', 'transformernn']:
            break
        print("Invalid choice. Please enter 'DeepNN' or 'TransformerNN'.")
    
    # Initialize global model
    if model_choice == 'deepnn':
        global_model = DeepNN(config.input_dim).to(config.device)
    else:
        global_model = TransformerNN(config.input_dim).to(config.device)
    
    # Filter clients and calculate quality scores
    eligible_clients = [i for i in range(num_clients) if client_class_dist[i].get(1, 0) >= 8]
    quality_scores = [utils.client_quality_score_enhanced(client_class_dist[i]) for i in eligible_clients]
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
    
    # Compute class weights for focal loss
    y_test_np = test_loader.dataset.tensors[1].numpy()
    class_weights = utils.compute_class_weights(y_test_np, config.scale_factor)
    focal_criterion = AdaptiveFocalLoss(alpha=class_weights, gamma_min=config.gamma_min, gamma_max=config.gamma_max)
    
    # Setup CSV file
    csv_file = f"fl_metrics_{model_choice.upper()}_{algo_choice.upper()}_epochs_{config.local_epochs}_frac_{config.fraction_min}_{config.fraction_max}_rounds_{config.num_rounds}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Avg_Train_Loss', 'Val_Loss', 'F1_Score', 'Accuracy', 'Balanced_Acc', 
                         'Confusion_Matrix', 'Balanced_F1', 'Balanced_Acc', 'Classification_Report', 'Model'])
    
    best_f1, best_balanced_acc = 0.0, 0.0
    
    # Training loop
    for round_num in range(1, config.num_rounds + 1):
        num_selected = max(1, int(random.uniform(config.fraction_min, config.fraction_max) * len(eligible_clients)))
        selected_clients = np.random.choice(eligible_clients, size=min(num_selected, len(eligible_clients)), 
                                            p=probabilities, replace=False)
        
        print(f"\nRound {round_num}: Selected {len(selected_clients)} clients")
        
        client_models = []
        train_losses = []
        
        for client_idx in selected_clients:
            if model_choice == 'deepnn':
                local_model = DeepNN(config.input_dim)
            else:
                local_model = TransformerNN(config.input_dim)
            local_model = local_model.to(config.device)
            local_model.load_state_dict(global_model.state_dict())
            loader, data_size = client_data[client_idx]
            
            if data_size < 64:
                continue
                
            client_quality = quality_scores[eligible_clients.index(client_idx)]
            
            if algo_choice == 'fedprox':
                loss = train_local_fedprox(local_model, global_model, loader, client_quality, 
                                          epochs=config.local_epochs, proximal_mu=config.proximal_mu)
                if loss == float('inf') or loss < 1e-4:
                    continue
                client_models.append((local_model.state_dict(), data_size, client_quality))
            else:  # FedNoLoWe
                optimizer = optim.Adam(local_model.parameters(), lr=utils.get_client_lr(client_quality), weight_decay=1e-4)
                local_model, loss = train_local_model_fednolowe(local_model, loader, focal_criterion, optimizer, 
                                                              epochs=config.local_epochs)
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
        
        # Evaluate
        val_loss, f1, cm, accuracy, balanced_acc, report_str = utils.evaluate(global_model, test_loader, config.device)
        _, balanced_f1, _, _, balanced_balanced_acc, _ = utils.evaluate(global_model, balanced_test_loader, config.device)
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
        
        # Save best model
        if balanced_f1 > best_f1:
            best_f1 = balanced_f1
            best_balanced_acc = balanced_balanced_acc
            torch.save(global_model.state_dict(), config.best_model_path)
            print(f"🎯 New best Balanced F1: {balanced_f1:.4f}, Balanced Acc: {balanced_balanced_acc:.4f}")
        else:
            print(f"⏳ No improvement in this round.")
    
    print(f"\n✅ FL completed after {config.num_rounds} rounds using {algo_choice.upper()} with {model_choice.upper()}.")
    
    # Final evaluation
    if os.path.exists(config.best_model_path):
        global_model.load_state_dict(torch.load(config.best_model_path, map_location=config.device))
        print(f"\n📊 Final evaluation with best model.")
    
    final_val_loss, final_f1, final_cm, final_accuracy, final_balanced_acc, final_report = utils.evaluate(global_model, test_loader, config.device)
    final_balanced_loss, final_balanced_f1, _, _, final_balanced_balanced_acc, _ = utils.evaluate(global_model, balanced_test_loader, config.device)
    
    print(f"\n🎉 Final Results with Best Model:")
    print(f"Test Loss: {final_val_loss:.4f}, F1: {final_f1:.4f}, Acc: {final_accuracy:.4f}, Balanced Acc: {final_balanced_acc:.4f}")
    print(f"Balanced Test Loss: {final_balanced_loss:.4f}, Balanced F1: {final_balanced_f1:.4f}, Balanced Balanced Acc: {final_balanced_balanced_acc:.4f}")
    print(f"Confusion Matrix:\n{final_cm}")
    print(f"Classification Report:\n{final_report}")
    
    print(f"\n✅ FL completed. Best Balanced F1: {best_f1:.4f}, Best Balanced Acc: {best_balanced_acc:.4f}")
    
    utils.save_predictions(global_model, test_loader, config.predictions_filename, config.device)

if __name__ == "__main__":
    main()