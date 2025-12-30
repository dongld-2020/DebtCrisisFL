import torch
import torch.nn as nn
import config

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, config.hidden_dim1)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim1)
        self.fc2 = nn.Linear(config.hidden_dim1, config.hidden_dim2)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim2)
        self.fc3 = nn.Linear(config.hidden_dim2, config.hidden_dim3)
        self.bn3 = nn.BatchNorm1d(config.hidden_dim3)
        self.fc4 = nn.Linear(config.hidden_dim3, config.output_dim)
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

class TransformerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=8, num_layers=3, dropout=0.5):
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
        self.fc = nn.Linear(hidden_dim, config.output_dim)
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