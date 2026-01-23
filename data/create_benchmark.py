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
    Đọc dữ liệu từ file Excel, tạo tập global test và phân chia dữ liệu còn lại
    cho từng client dựa trên mã quốc gia. Xuất dữ liệu ra file Excel mới.

    Args:
        input_file (str): Tên file Excel (.xlsx) đầu vào.
        output_file (str): Tên file Excel (.xlsx) đầu ra.
        test_size (float): Tỷ lệ dữ liệu cho tập global test.

    Returns:
        tuple: (global_test_data, client_data_dict, label_column)
    """
    try:
        # 1. Đọc và tiền xử lý dữ liệu
        df = pd.read_excel(input_file)
        df = df.replace(['no data', 'No data'], 0).fillna(0)
        
        # Kiểm tra cột 'Code' và số cột
        if 'Code' not in df.columns:
            raise ValueError("Excel file phải chứa cột 'Code'.")
        if df.shape[1] < 4:
            raise ValueError("Excel file phải có ít nhất 4 cột: Code, 1 cột khác, ≥1 đặc trưng, và nhãn.")
        
        # 2. Tách nhãn và đặc trưng
        label_column = df.columns[-1]
        print(f"Cột nhãn được chọn: {label_column}")
        if label_column not in df.columns:
            raise ValueError(f"Cột nhãn '{label_column}' không tồn tại trong dữ liệu.")
        
        y = df[label_column]
        features = df.iloc[:, 2:-1]
        if features.empty:
            raise ValueError("Không có cột đặc trưng nào được chọn.")

        # 3. Tạo tập kiểm thử toàn cầu
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=test_size, random_state=42
        )
        feature_columns = features.columns
        global_test_data = pd.DataFrame(X_test, columns=feature_columns)
        global_test_data[label_column] = y_test
        
        # Chuẩn hóa dữ liệu đặc trưng
        min_vals = global_test_data[feature_columns].min()
        max_vals = global_test_data[feature_columns].max()
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        global_test_data[feature_columns] = (global_test_data[feature_columns] - min_vals) / range_vals

        # 4. Phân chia dữ liệu cho từng client
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

        # 5. Xuất dữ liệu ra file Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            global_test_data.to_excel(writer, sheet_name='Global_Test', index=False)
            for client_id, client_df in client_data_dict.items():
                client_df.to_excel(writer, sheet_name=client_id, index=False)

        print(f"Đã xuất dữ liệu ra file '{output_file}'.")

        return global_test_data, client_data_dict, label_column

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_file}'. Vui lòng kiểm tra lại.")
        return None, None, None
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return None, None, None

def plot_stacked_class_distribution(client_data, label_column, save_path):
    """Vẽ biểu đồ stacked bar cho phân bố lớp."""
    try:
        class_dist = {}
        valid_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
        if not valid_dfs:
            print("Không có client nào có cột nhãn hợp lệ để vẽ stacked bar.")
            return
        
        unique_classes = sorted(pd.concat(valid_dfs)[label_column].unique())
        
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Cảnh báo: Client '{client_id}' không có cột '{label_column}'. Bỏ qua.")
                continue
            counts = client_df[label_column].value_counts(normalize=True) * 100
            class_dist[client_id] = [counts.get(cls, 0) for cls in unique_classes]
        
        if not class_dist:
            print("Không có dữ liệu phân bố lớp để vẽ stacked bar.")
            return
        
        dist_df = pd.DataFrame(class_dist, index=[f'Class {cls}' for cls in unique_classes]).T
        
        fig, ax = plt.subplots(figsize=(12, 8))
        dist_df.plot(kind='bar', stacked=True, ax=ax, cmap='viridis')
        ax.set_title('Phân Bố Lớp Theo Từng Client (Label Skew)')
        ax.set_xlabel('Client (Mã Quốc Gia)')
        ax.set_ylabel('Tỷ Lệ Phần Trăm (%)')
        ax.legend(title='Lớp', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)  # Đóng figure để tránh lỗi GUI
        if os.path.exists(save_path):
            print(f"Đã lưu biểu đồ stacked bar tại '{save_path}'.")
        else:
            print(f"Lỗi: Không thể lưu biểu đồ stacked bar tại '{save_path}'.")
    except Exception as e:
        print(f"Lỗi khi vẽ stacked bar: {e}")

def plot_heatmap_class_distribution(client_data, label_column, save_path):
    """Vẽ heatmap cho số lượng mẫu theo lớp."""
    try:
        class_dist = {}
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Cảnh báo: Client '{client_id}' không có cột '{label_column}'. Bỏ qua.")
                continue
            counts = client_df[label_column].value_counts()
            class_dist[client_id] = counts.to_dict()
        
        if not class_dist:
            print("Không có dữ liệu phân bố lớp để vẽ heatmap.")
            return
        
        dist_df = pd.DataFrame(class_dist).T.fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, len(client_data) * 0.5 + 2))
        sns.heatmap(dist_df, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
        ax.set_title('Heatmap Số Lượng Mẫu Theo Lớp (Quantity & Label Skew)')
        ax.set_xlabel('Lớp')
        ax.set_ylabel('Client (Mã Quốc Gia)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        if os.path.exists(save_path):
            print(f"Đã lưu heatmap tại '{save_path}'.")
        else:
            print(f"Lỗi: Không thể lưu heatmap tại '{save_path}'.")
    except Exception as e:
        print(f"Lỗi khi vẽ heatmap: {e}")

def plot_feature_histogram(client_data, feature_name, selected_clients=None, save_path='figures/feature_histogram.png'):
    """Vẽ histogram cho một feature giữa các client."""
    try:
        if selected_clients is None:
            selected_clients = list(client_data.keys())[:5]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for client_id in selected_clients:
            client_df = client_data.get(client_id)
            if client_df is None or feature_name not in client_df.columns:
                print(f"Cảnh báo: Client '{client_id}' không có hoặc thiếu feature '{feature_name}'. Bỏ qua.")
                continue
            sns.histplot(client_df[feature_name], kde=True, label=client_id, ax=ax)
            plotted = True
        
        if not plotted:
            print("Không có dữ liệu để vẽ histogram.")
            return
        
        ax.set_title(f'Phân Bố Feature "{feature_name}" (Feature Skew)')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Tần Suất')
        ax.legend(title='Client')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        if os.path.exists(save_path):
            print(f"Đã lưu histogram tại '{save_path}'.")
        else:
            print(f"Lỗi: Không thể lưu histogram tại '{save_path}'.")
    except Exception as e:
        print(f"Lỗi khi vẽ histogram: {e}")

def plot_feature_boxplot(client_data, feature_name, save_path):
    """Vẽ boxplot cho một feature giữa các client."""
    try:
        data = []
        labels = []
        for client_id, client_df in client_data.items():
            if feature_name not in client_df.columns:
                print(f"Cảnh báo: Client '{client_id}' không có feature '{feature_name}'. Bỏ qua.")
                continue
            data.append(client_df[feature_name])
            labels.append(client_id)
        
        if not data:
            print("Không có dữ liệu để vẽ boxplot.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(data, labels=labels)
        ax.set_title(f'Boxplot của Feature "{feature_name}" (Feature Skew)')
        ax.set_xlabel('Client (Mã Quốc Gia)')
        ax.set_ylabel(feature_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        if os.path.exists(save_path):
            print(f"Đã lưu boxplot tại '{save_path}'.")
        else:
            print(f"Lỗi: Không thể lưu boxplot tại '{save_path}'.")
    except Exception as e:
        print(f"Lỗi khi vẽ boxplot: {e}")

def generate_summary_table(client_data, label_column, save_path):
    """Tạo bảng tóm tắt phân bố lớp, xuất ra LaTeX."""
    try:
        summary_data = []
        valid_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
        if not valid_dfs:
            print("Không có client nào có cột nhãn hợp lệ để tạo bảng.")
            return None
        
        unique_classes = sorted(pd.concat(valid_dfs)[label_column].unique())
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Cảnh báo: Client '{client_id}' không có cột '{label_column}'. Bỏ qua.")
                continue
            total_samples = len(client_df)
            class_counts = client_df[label_column].value_counts(normalize=True) * 100
            row = [client_id, total_samples] + [f"{class_counts.get(cls, 0):.2f}" for cls in unique_classes]
            summary_data.append(row)
        
        if not summary_data:
            print("Không có dữ liệu để tạo bảng tóm tắt.")
            return None
        
        columns = ['Client ID', 'Total Samples'] + [f'\% Class {cls}' for cls in unique_classes]
        summary_df = pd.DataFrame(summary_data, columns=columns)
        
        latex_table = summary_df.to_latex(
            index=False,
            caption='Tóm Tắt Phân Bố Lớp Theo Client (Minh Họa Extreme Non-IID)',
            label='tab:client_distribution',
            column_format='|l|r|' + 'r|' * (len(columns) - 2),
            escape=False,
            float_format="%.2f"
        )
        with open(save_path, 'w') as f:
            f.write(latex_table)
        if os.path.exists(save_path):
            print(f"Đã lưu bảng LaTeX tại '{save_path}'.")
        else:
            print(f"Lỗi: Không thể lưu bảng LaTeX tại '{save_path}'.")
        return summary_df
    except Exception as e:
        print(f"Lỗi khi tạo bảng tóm tắt: {e}")
        return None

def calculate_kl_divergence(client_data, label_column):
    """Tính KL Divergence giữa các client."""
    try:
        class_dist = {}
        valid_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
        if not valid_dfs:
            print("Không có client nào có cột nhãn hợp lệ để tính KL Divergence.")
            return {}
        
        unique_classes = sorted(pd.concat(valid_dfs)[label_column].unique())
        if not unique_classes:
            print("Không tìm thấy lớp nào để tính KL Divergence.")
            return {}
        
        for client_id, client_df in client_data.items():
            if label_column not in client_df.columns:
                print(f"Cảnh báo: Client '{client_id}' không có cột '{label_column}'. Bỏ qua.")
                continue
            counts = client_df[label_column].value_counts(normalize=True)
            class_dist[client_id] = counts.reindex(unique_classes, fill_value=1e-10).values
        
        if len(class_dist) < 2:
            print("Cần ít nhất 2 client để tính KL Divergence.")
            return {}
        
        kl_divs = {}
        for client1, dist1 in class_dist.items():
            for client2, dist2 in class_dist.items():
                if client1 < client2:
                    kl = entropy(dist1, dist2)
                    kl_divs[f'{client1} vs {client2}'] = kl
        return kl_divs
    except Exception as e:
        print(f"Lỗi khi tính KL Divergence: {e}")
        return {}

# --- Khởi tạo và chạy chương trình ---

input_file_name = "data_raw.xlsx"
output_file_name = "data_benchmark.xlsx"

# Sử dụng đường dẫn tuyệt đối cho thư mục figures
base_dir = os.path.dirname(os.path.abspath(__file__))  # Thư mục của script
figures_dir = os.path.join(base_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Đã tạo thư mục '{figures_dir}'.")

global_test, client_data, label_column = create_fl_clients(input_file_name, output_file_name)

if global_test is not None and client_data is not None:
    # Thống kê cơ bản
    print(f"\nTổng số mẫu dữ liệu: {len(global_test) + sum(len(df) for df in client_data.values())}")
    print(f"Tập kiểm thử toàn cầu: {len(global_test)} mẫu")
    print(f"Số client: {len(client_data)}")
    
    # Thống kê từng client
    print(f"\n--- THỐNG KÊ TỪNG CLIENT ---")
    for client_id, client_df in client_data.items():
        print(f"\nClient '{client_id}':")
        print(f"  - Tổng số mẫu: {len(client_df)}")
        if label_column in client_df.columns:
            print("  - Phân bố lớp:")
            class_counts = client_df[label_column].value_counts()
            for class_label, count in class_counts.items():
                percentage = count / len(client_df) * 100 if len(client_df) > 0 else 0
                print(f"    - Lớp {class_label}: {count} mẫu ({percentage:.2f}%)")
        else:
            print(f"  - Lỗi: Không tìm thấy cột '{label_column}' trong client '{client_id}'.")
    
    # Thống kê tổng thể
    print(f"\n--- THỐNG KÊ TỔNG THỂ ---")
    all_client_samples = sum(len(df) for df in client_data.values())
    print(f"Tổng số mẫu trong tất cả clients: {all_client_samples}")
    valid_client_dfs = [df[label_column] for df in client_data.values() if label_column in df.columns]
    if valid_client_dfs:
        all_client_labels = pd.concat(valid_client_dfs)
        print("Phân bố lớp tổng thể:")
        total_class_counts = all_client_labels.value_counts()
        for class_label, count in total_class_counts.items():
            percentage = count / all_client_samples * 100 if all_client_samples > 0 else 0
            print(f"  - Lớp {class_label}: {count} mẫu ({percentage:.2f}%)")
    
    # Thống kê tập global test
    print(f"\n--- THỐNG KÊ TẬP GLOBAL TEST ---")
    print(f"Tổng số mẫu: {len(global_test)}")
    if label_column in global_test.columns:
        print("Phân bố lớp:")
        class_counts = global_test[label_column].value_counts()
        for class_label, count in class_counts.items():
            print(f"  - Lớp {class_label}: {count} mẫu ({count/len(global_test)*100:.2f}%)")
    
    # Tính KL Divergence
    print(f"\n--- KL DIVERGENCE GIỮA CÁC CLIENT ---")
    kl_divs = calculate_kl_divergence(client_data, label_column)
    for pair, kl in kl_divs.items():
        print(f"KL Divergence giữa {pair}: {kl:.4f}")

    # Vẽ biểu đồ với đường dẫn tuyệt đối
    print("\n--- VẼ BIỂU ĐỒ MINH HỌA NON-IID ---")
    plot_stacked_class_distribution(client_data, label_column, os.path.join(figures_dir, 'stacked_class_dist.png'))
    plot_heatmap_class_distribution(client_data, label_column, os.path.join(figures_dir, 'heatmap_class_dist.png'))
    
    feature_columns = list(client_data[list(client_data.keys())[0]].columns[:-1])
    if feature_columns:
        plot_feature_histogram(client_data, feature_columns[0], save_path=os.path.join(figures_dir, 'feature_histogram.png'))
        plot_feature_boxplot(client_data, feature_columns[0], os.path.join(figures_dir, 'feature_boxplot.png'))
    
    # Tạo bảng tóm tắt
    print("\n--- TẠO BẢNG TÓM TẮT ---")
    summary_df = generate_summary_table(client_data, label_column, os.path.join(figures_dir, 'summary_table.tex'))
    if summary_df is not None:
        print("\nBảng tóm tắt:")
        print(summary_df)