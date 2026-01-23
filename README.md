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
🚀 InstallationClone the repository:Bashgit clone [https://github.com/dongld-2020/DebtCrisisFL.git](https://github.com/dongld-2020/DebtCrisisFL.git)
cd DebtCrisisFL
Install dependencies:Ensure you have Python installed, then install the required libraries:Bashpip install pandas numpy torch scikit-learn openpyxl
🛠️ Usage1. Run a Single ExperimentTo run a standard Federated Learning experiment with interactive model selection:Bashpython run_experiment.py
Interactive Mode: You will be prompted to select a model (e.g., DeepNN, TransformerNN) and an algorithm (FedProx or FedNoLoWe).Output: Metrics (F1, Accuracy, Balanced Accuracy) are displayed in the console and saved to a CSV file.2. Run Grid SearchTo perform hyperparameter tuning (Grid Search) to find the optimal values for proximal_mu, gamma, and scale_factor:Bashpython run_experinemt_grid_search.py
Note: This process iterates through multiple combinations of hyperparameters and may take a significant amount of time.Output: Saves a detailed CSV report of all parameter combinations and identifies the best configuration.📊 Methodology HighlightsAdaptive Focal Loss FormulaThe loss function evolves over time to combat class imbalance:$$FL(p_t) = -\alpha_t (1 - p_t)^{\gamma_t} \log(p_t)$$Where $\gamma_t$ is the focusing parameter at step $t$:$$\gamma_t = \gamma_{min} + (\gamma_{max} - \gamma_{min}) \times \frac{current\_step}{total\_steps}$$FedNoLoWe AggregationUnlike standard FedAvg which weights clients purely by data size, FedNoLoWe calculates weights ($w_k$) based on client training loss ($L_k$):Normalize losses across clients.Apply inverse weighting (lower loss = higher weight).(Optional) Apply Softmax to ensure probability distribution.📝 OutputsThe system automatically generates:Training Logs: CSV files tracking loss and accuracy per round.Confusion Matrices: To visualize True Positives vs. False Negatives.Prediction Files: predictions_analysis.csv containing raw probabilities for further analysis.🤝 ContributingContributions are welcome! Please feel free to submit a Pull Request or open an issue for discussion.📧 ContactProject Link: https://github.com/dongld-2020/DebtCrisisFL
