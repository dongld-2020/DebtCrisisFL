# DebtCrisisFL: Federated Learning for Sovereign Debt Crisis Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**DebtCrisisFL** is a research framework designed to predict sovereign debt crises using Federated Learning (FL). This project addresses the challenges of **imbalanced data** in financial datasets by implementing advanced techniques such as **Adaptive Focal Loss** and a novel aggregation strategy called **FedNoLoWe** (Federated Normalized Loss Weighted).

## 📊 Dataset Information
The dataset is derived from the IMF and World Bank. It includes quarterly time-series data.
- **File:** `data/raw_data.xlsx`
- **Samples:** 24,574 observations.
- **Features:** 26 macroeconomic indicators (e.g., GDP Growth, External Debt, FDI).
- **Target:** `CrisisIndexTotal` (Binary classification: 0 = No Crisis, 1 = Crisis).
- **Split:** 80% Training (distributed across 266 clients/countries), 20% Global Testing.
- 
## 🌟 Key Features

* **Algorithms:**
    * **FedProx:** Implements the proximal term to handle system heterogeneity.
    * **FedNoLoWe (Novel):** A custom aggregation strategy that weights client updates based on their normalized training loss (inverse loss weighting).
* **Imbalanced Data Handling:**
    * **Adaptive Focal Loss:** Dynamically adjusts the focusing parameter ($\gamma$) during training to focus on hard-to-classify examples (minority class).
    * **Enhanced Client Quality Scoring:** assessing clients based on their data balance ratio.
* **Models:** Supports multiple architectures including:
    * Deep Neural Network (DeepNN)
    * Transformer Neural Network
    * 1D CNN (CNN1D)
    * Residual Neural Network (ResNet)
    * Simple Feedforward Neural Network (FNN)
* **Grid Search:** Built-in script for hyperparameter optimization.

## 📂 Project Structure

```bash
DebtCrisisFL/
├── data/
│   ├── raw_data.xlsx            # Main dataset (Excel format)
│   ├── data_benchmark.xlsx      # Benchmark data
│   ├── create_benchmark.py      # Script to create benchmark data from raw data
│   └── discriptive_raw.py       # Script for data descriptive analysis
├── outcomes/                    # Directory to store experiment results and logs
├── run_experiment.py            # Main script to run a single FL experiment
├── run_experinemt_grid_search.py # Script for Grid Search (Hyperparameter tuning)
└── README.md                    # Project documentation
```
## 🚀 Installation

**1. Clone the repository:**
git clone [https://github.com/dongld-2020/DebtCrisisFL.git](https://github.com/dongld-2020/DebtCrisisFL.git)
cd DebtCrisisFL

**2. Install dependencies:** Ensure you have Python installed, then install the required libraries:
```bash
pip install pandas numpy torch scikit-learn openpyxl
```

## 🛠️ Usage
**1. Run a Single Experiment**
To run a standard Federated Learning experiment with interactive model selection:
```bash
python run_experiment.py
```
You will be prompted to choose a model (e.g., DeepNN, TransformerNN) and an algorithm (FedProx or FedNoLoWe).

Results (metrics, confusion matrix) will be saved to a CSV file.

**2. Run Grid Search**
To perform hyperparameter tuning (Grid Search) to find the best proximal_mu, gamma, and scale_factor:
```bash
python run_experinemt_grid_search.py
```
## 🧠 Methodology

**FedNoLoWe** 

(Federated Normalized Loss Weighted) Unlike standard FedAvg which aggregates based on dataset size, FedNoLoWe assigns aggregation weights based on the model's performance on local data. Clients with lower training loss (indicating better learning) contribute more to the global model using a Softmax-based or normalized inverse weighting mechanism.

**Adaptive Focal Loss**

The loss function evolves over time:

$$ FL(p_t) = -\alpha_t (1 - p_t)^{\gamma_t} \log(p_t) $$

Where $\gamma_t$ increases linearly from $\gamma_{min}$ to $\gamma_{max}$ as training progresses, forcing the model to focus harder on difficult samples in later rounds.
