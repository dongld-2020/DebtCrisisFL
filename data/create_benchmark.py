import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
import pathlib

def create_fl_clients(input_file, output_file, test_size=0.2):
    """
    Read data from an Excel file, create a global test set, and distribute the remaining data
    to each client based on country code. Export the data to a new Excel file.

    Args:
        input_file (str): Input Excel filename (.xlsx).
        output_file (str): Output Excel filename (.xlsx).
        test_size (float): Data ratio for the global test set.

    Returns:
        tuple: (global_test_data, client_data_dict, label_column)
    """
    try:
        #1. Reading and preprocessing data
        df = pd.read_excel(input_file)
        df = df.replace(['no data', 'No data'], 0).fillna(0)
        
        # Check the 'Code' column and the number of columns
        if 'Code' not in df.columns:
            raise ValueError("The Excel file must contain a 'Code' column.")
        if df.shape[1] < 4:
            raise ValueError("An Excel file must have at least four columns: Code, one other column, at least one feature, and a label.")
        
        #2. Separating labels and features
        label_column = df.columns[-1]
        print(f"Selected label column: {label_column}")
        if label_column not in df.columns:
            raise ValueError(f"Lable column '{label_column}' does not exist in the data.")
        
        y = df[label_column]
        features = df.iloc[:, 2:-1]
        if features.empty:
            raise ValueError("No feature columns were selected.")

        #3. Create a global test set
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=test_size, random_state=42
        )
        feature_columns = features.columns
        global_test_data = pd.DataFrame(X_test, columns=feature_columns)
        global_test_data[label_column] = y_test
        
        # Normalizing feature data
        min_vals = global_test_data[feature_columns].min()
        max_vals = global_test_data[feature_columns].max()
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        global_test_data[feature_columns] = (global_test_data[feature_columns] - min_vals) / range_vals

        #4. Divide the data among each client.
        train_indices = X_train.index
        train_df = df.loc[train_indices].copy()
        client_data_dict = {}
        grouped_by_client = train_df.groupby('Code')
        
        for client_id, client_df in grouped_by_client:
            client_X = client_df.iloc[:, 2:-1]
            client_y = client_df[label_column]
            client_min_vals = client_X.min()
            client_max_vals = client_X.max()
            client_range_vals = client_max_vals - client_min_vals
            client_range_vals[client_range_vals == 0] = 1
            client_X_normalized = (client_X - client_min_vals) / client_range_vals
            client_X_normalized = client_X_normalized.fillna(0)
            client_full_df = client_X_normalized.copy()
            client_full_df[label_column] = client_y
            client_data_dict[client_id] = client_full_df
            print(f"Client '{client_id}': {len(client_df)} mẫu, cột: {list(client_full_df.columns)}")

        #5. Export data to an Excel file
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            global_test_data.to_excel(writer, sheet_name='Global_Test', index=False)
            for client_id, client_df in client_data_dict.items():
                client_df.to_excel(writer, sheet_name=client_id, index=False)

        print(f"Data has been exported to a file. '{output_file}'.")

        return global_test_data, client_data_dict, label_column

    except FileNotFoundError:
        print(f"Error: File not found '{input_file}'. Please check.")
        return None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

def plot_stacked_class_distribution(client_data, label_column, save_path):
    """Draw a stacked bar chart for the class distribution."""
    try:
        class_dist = {}
        valid_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
        if not valid_dfs:
            print("No client has a valid label column to draw a stacked bar.")
            return
        
        unique_classes = sorted(pd.concat(valid_dfs)[label_column].unique())
        
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Warning: Client '{client_id}' dose not have '{label_column}'.")
                continue
            counts = client_df[label_column].value_counts(normalize=True) * 100
            class_dist[client_id] = [counts.get(cls, 0) for cls in unique_classes]
        
        if not class_dist:
            print("There is no class distribution data available to draw stacked bars.")
            return
        
        dist_df = pd.DataFrame(class_dist, index=[f'Class {cls}' for cls in unique_classes]).T
        
        fig, ax = plt.subplots(figsize=(12, 8))
        dist_df.plot(kind='bar', stacked=True, ax=ax, cmap='viridis')
        ax.set_title('Layer Distribution by Client (Label Skew)')
        ax.set_xlabel('Client (Country code)')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to avoid GUI errors.
        if os.path.exists(save_path):
            print(f"Stacked bar chart saved at '{save_path}'.")
        else:
            print(f"Error: Cannot save stacked bar chart at '{save_path}'.")
    except Exception as e:
        print(f"Error when drawing stacked bars: {e}")

def plot_heatmap_class_distribution(client_data, label_column, save_path):
    """Draw a heatmap for the number of samples by layer."""
    try:
        class_dist = {}
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Warning: Client '{client_id}' dose not have '{label_column}'. ")
                continue
            counts = client_df[label_column].value_counts()
            class_dist[client_id] = counts.to_dict()
        
        if not class_dist:
            print("There is no layer distribution data available to draw a heatmap.")
            return
        
        dist_df = pd.DataFrame(class_dist).T.fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, len(client_data) * 0.5 + 2))
        sns.heatmap(dist_df, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
        ax.set_title('Heatmap of Sample Quantity by Layer (Quantity & Label Skew)')
        ax.set_xlabel('Class')
        ax.set_ylabel('Client (Country code)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        if os.path.exists(save_path):
            print(f"Save heatmap at '{save_path}'.")
        else:
            print(f"Error: Can not save heatmap at '{save_path}'.")
    except Exception as e:
        print(f"Error when draw heatmap: {e}")

def plot_feature_histogram(client_data, feature_name, selected_clients=None, save_path='figures/feature_histogram.png'):
    """Draw a histogram for a feature between clients."""
    try:
        if selected_clients is None:
            selected_clients = list(client_data.keys())[:5]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for client_id in selected_clients:
            client_df = client_data.get(client_id)
            if client_df is None or feature_name not in client_df.columns:
                print(f"Warning: Client '{client_id}' absent or missing feature '{feature_name}'.")
                continue
            sns.histplot(client_df[feature_name], kde=True, label=client_id, ax=ax)
            plotted = True
        
        if not plotted:
            print("Do not have data to draw histogram.")
            return
        
        ax.set_title(f'Feature distribution"{feature_name}" (Feature Skew)')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frequency')
        ax.legend(title='Client')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        if os.path.exists(save_path):
            print(f"Save histogram at '{save_path}'.")
        else:
            print(f"Error: Can not save histogram at '{save_path}'.")
    except Exception as e:
        print(f"Error when drawing histogram: {e}")

def plot_feature_boxplot(client_data, feature_name, save_path):
    """Draw a boxplot for a feature between clients."""
    try:
        data = []
        labels = []
        for client_id, client_df in client_data.items():
            if feature_name not in client_df.columns:
                print(f"Warning: Client '{client_id}' does not feature '{feature_name}'.")
                continue
            data.append(client_df[feature_name])
            labels.append(client_id)
        
        if not data:
            print("Do not have data to draw boxplot.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(data, labels=labels)
        ax.set_title(f'Boxplot of Feature "{feature_name}" (Feature Skew)')
        ax.set_xlabel('Client (Country code)')
        ax.set_ylabel(feature_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        if os.path.exists(save_path):
            print(f"Save boxplot at '{save_path}'.")
        else:
            print(f"Error: Can not save boxplot at '{save_path}'.")
    except Exception as e:
        print(f"Error when drawing boxplot: {e}")

def generate_summary_table(client_data, label_column, save_path):
    """Create a class distribution summary table and export it to LaTeX."""
    try:
        summary_data = []
        valid_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
        if not valid_dfs:
            print("No client has a valid label column to create the table.")
            return None
        
        unique_classes = sorted(pd.concat(valid_dfs)[label_column].unique())
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Warning: Client '{client_id}' does not have column '{label_column}'.")
                continue
            total_samples = len(client_df)
            class_counts = client_df[label_column].value_counts(normalize=True) * 100
            row = [client_id, total_samples] + [f"{class_counts.get(cls, 0):.2f}" for cls in unique_classes]
            summary_data.append(row)
        
        if not summary_data:
            print("There is no data to generate a summary table.")
            return None
        
        columns = ['Client ID', 'Total Samples'] + [f'\% Class {cls}' for cls in unique_classes]
        summary_df = pd.DataFrame(summary_data, columns=columns)
        
        latex_table = summary_df.to_latex(
            index=False,
            caption='Summary of Client-Based Class Distribution (Extreme Non-IID Illustration)',
            label='tab:client_distribution',
            column_format='|l|r|' + 'r|' * (len(columns) - 2),
            escape=False,
            float_format="%.2f"
        )
        with open(save_path, 'w') as f:
            f.write(latex_table)
        if os.path.exists(save_path):
            print(f"Save table LaTeX at '{save_path}'.")
        else:
            print(f"Error: Can not save table LaTeX at '{save_path}'.")
        return summary_df
    except Exception as e:
        print(f"Error when creating summary table: {e}")
        return None

def calculate_kl_divergence(client_data, label_column):
    """Calculate the KL Divergence between clients."""
    try:
        class_dist = {}
        valid_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
        if not valid_dfs:
            print("No client has a valid label column to calculate KL Divergence.")
            return {}
        
        unique_classes = sorted(pd.concat(valid_dfs)[label_column].unique())
        if not unique_classes:
            print("No class was found to calculate KL Divergence.")
            return {}
        
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Warning: Client '{client_id}' does not have column '{label_column}'.")
                continue
            counts = client_df[label_column].value_counts(normalize=True)
            class_dist[client_id] = counts.reindex(unique_classes, fill_value=1e-10).values
        
        if len(class_dist) < 2:
            print("At least two clients are needed to calculate KL Divergence.")
            return {}
        
        kl_divs = {}
        for client1, dist1 in class_dist.items():
            for client2, dist2 in class_dist.items():
                if client1 < client2:
                    kl = entropy(dist1, dist2)
                    kl_divs[f'{client1} vs {client2}'] = kl
        return kl_divs
    except Exception as e:
        print(f"Error in calculating KL Divergence: {e}")
        return {}

# --- Initialize and run the program ---

input_file_name = "data_raw.xlsx"
output_file_name = "data_benchmark.xlsx"

Use absolute paths for the figures folder.
base_dir = os.path.dirname(os.path.abspath(__file__))  
figures_dir = os.path.join(base_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Folder created '{figures_dir}'.")

global_test, client_data, label_column = create_fl_clients(input_file_name, output_file_name)

if global_test is not None and client_data is not None:
    #Basic Statistics
    print(f"\nTotal number of data samples: {len(global_test) + sum(len(df) for df in client_data.values())}")
    print(f"Global test suite: {len(global_test)} samples")
    print(f"Number of client: {len(client_data)}")
    
    # Client-specific statistics
    print(f"\n--- Client-specific statistics ---")
    for client_id, client_df in client_data.items():
        print(f"\nClient '{client_id}':")
        print(f"  - Total samples: {len(client_df)}")
        if label_column in client_df.columns:
            print("  - Class distribution:")
            class_counts = client_df[label_column].value_counts()
            for class_label, count in class_counts.items():
                percentage = count / len(client_df) * 100 if len(client_df) > 0 else 0
                print(f"    - Class {class_label}: {count} samples ({percentage:.2f}%)")
        else:
            print(f"  - Error: Can not find column '{label_column}' in client '{client_id}'.")
    
    #Overall Statistics
    print(f"\n--- Overall Statistics ---")
    all_client_samples = sum(len(df) for df in client_data.values())
    print(f"Total samples in all clients: {all_client_samples}")
    valid_client_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
    if valid_client_dfs:
        all_client_labels = pd.concat(valid_client_dfs)
        print("Overall class distribution:")
        total_class_counts = all_client_labels.value_counts()
        for class_label, count in total_class_counts.items():
            percentage = count / all_client_samples * 100 if all_client_samples > 0 else 0
            print(f"  - Class {class_label}: {count} samples ({percentage:.2f}%)")
    
    # Statistics of the global test set
    print(f"\n--- Statistics of the global test set ---")
    print(f"Total samples: {len(global_test)}")
    if label_column in global_test.columns:
        print("Class distribution:")
        class_counts = global_test[label_column].value_counts()
        for class_label, count in class_counts.items():
            print(f"  - Class {class_label}: {count} samples ({count/len(global_test)*100:.2f}%)")
    
    # Calculate KL Divergence
    print(f"\n--- KL DIVERGENCE BETWEEN CLIENT ---")
    kl_divs = calculate_kl_divergence(client_data, label_column)
    for pair, kl in kl_divs.items():
        print(f"KL Divergence between {pair}: {kl:.4f}")

    # Drawing graphs with absolute paths
    print("\n--- Drawing graphs with absolute paths ---")
    plot_stacked_class_distribution(client_data, label_column, os.path.join(figures_dir, 'stacked_class_dist.png'))
    plot_heatmap_class_distribution(client_data, label_column, os.path.join(figures_dir, 'heatmap_class_dist.png'))
    
    feature_columns = list(client_data[list(client_data.keys())[0]].columns[:-1])
    if feature_columns:
        plot_feature_histogram(client_data, feature_columns[0], save_path=os.path.join(figures_dir, 'feature_histogram.png'))
        plot_feature_boxplot(client_data, feature_columns[0], os.path.join(figures_dir, 'feature_boxplot.png'))
    
    # Create a summary table
    print("\n--- Create a summary table ---")
    summary_df = generate_summary_table(client_data, label_column, os.path.join(figures_dir, 'summary_table.tex'))
    if summary_df is not None:
        print("\nSummary table:")
        print(summary_df)