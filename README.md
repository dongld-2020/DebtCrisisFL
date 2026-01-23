# DebtCrisisFL: Federated Learning for Sovereign Debt Crisis Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**DebtCrisisFL** is a research framework designed to predict sovereign debt crises using Federated Learning (FL). This project addresses the challenges of **imbalanced data** in financial datasets by implementing advanced techniques such as **Adaptive Focal Loss** and a novel aggregation strategy called **FedNoLoWe** (Federated Normalized Loss Weighted).

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
