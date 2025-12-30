import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import config
import utils
from model import AdaptiveFocalLoss

def train_local_fedprox(model, global_model, loader, client_quality, epochs=5, proximal_mu=0.1, device=config.device):
    lr = utils.get_client_lr(client_quality, config.base_lr)
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
            
            criterion = AdaptiveFocalLoss(alpha=None, gamma_min=config.gamma_min, gamma_max=config.gamma_max)
            loss_standard = criterion(output, target.long())
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

def train_local_model_fednolowe(model, dataloader, criterion, optimizer, epochs=1, device=config.device):
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