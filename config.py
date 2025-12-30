import torch

# Set random seed
random_seed = 42

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File path
file_path = 'fl_data_exported_14092025.xlsx'

# Model parameters
input_dim = None  # Will be set dynamically
hidden_dim1, hidden_dim2, hidden_dim3 = 256, 128, 64
output_dim = 2

# Training parameters
num_rounds = 50
fraction_min, fraction_max = 0.3, 0.4
local_epochs = 10
proximal_mu = 0.5
base_lr = 0.001
min_samples = 10

# Focal loss parameters
gamma_min = 1.0
gamma_max = 4.5
scale_factor = 3.0

# File names
best_model_path = 'best_global_model_enhanced.pth'
predictions_filename = 'predictions_analysis.csv'