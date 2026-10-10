"""
run_grid_search.py
======================
Chạy lại grid search FedProx / FedNoLoWe theo protocol ĐÃ SỬA rò rỉ:

  - Class weight cho Adaptive Focal Loss: tính từ client_train_class_counts.csv
    (tổng hợp TRAIN, federated-safe), KHÔNG dùng nhãn của Global_Test.
  - Chọn hyperparameter / round tốt nhất: dựa trên Global_Val.
  - Global_Test: CHỈ đánh giá đúng 1 lần cho cấu hình đã chọn, không dùng
    để early-stop hay chọn hyperparameter.
  - Metric CHÍNH báo cáo trên phân phối mất cân bằng tự nhiên của Test
    (theo yêu cầu Editor); balanced-resampled Test chỉ là metric PHỤ.
  - Lặp lại N_SEEDS lần cho mỗi cấu hình tốt nhất/đã chọn để báo cáo
    mean +/- std (yêu cầu R3/R4/R6).

INPUT BẮT BUỘC (sinh ra bởi prepare_debtcrisisfl.py):
  - data_benchmark.xlsx  (sheet Global_Val, Global_Test, 1 sheet/client)
  - client_train_class_counts.csv

Cách chạy:
  python run_grid_search.py --model residualnn --algo fedprox
  python run_grid_search.py --model residualnn --algo fednolowe --n_seeds 5

Vì grid đầy đủ (60 cấu hình FedProx + 12 FedNoLoWe) x 3 kiến trúc x nhiều
seed rất tốn thời gian trên máy không có GPU, mặc định script chỉ chạy
REDUCED_GRID (xem bên dưới) -- có thể đổi lại FULL_GRID=True để chạy như
bản gốc.
"""

import argparse
import csv
import itertools
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import fl_common as flc

DATA_FILE = "data_benchmark_MISSING_STRATEGY_mean.xlsx"
COUNTS_FILE = "client_train_class_counts.csv"

FULL_GRID = True  # True = 60 (FedProx) / 12 (FedNoLoWe) tổ hợp như bản gốc

NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
FRACTION_MIN, FRACTION_MAX = 0.3, 0.4
BATCH_SIZE = 64
MIN_MINORITY_SAMPLES_TO_PARTICIPATE = 8  # xem ghi chú ablation bên dưới


def load_data():
    xl = pd.ExcelFile(DATA_FILE)
    val_df = pd.read_excel(xl, "Global_Val")
    test_df = pd.read_excel(xl, "Global_Test")
    client_sheets = [s for s in xl.sheet_names if s not in ("Global_Val", "Global_Test")]

    X_val = val_df.iloc[:, :-1].values.astype(np.float32)
    y_val = val_df.iloc[:, -1].values.astype(np.int64)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)

    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                             batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                              batch_size=BATCH_SIZE, shuffle=False)
    balanced_test_loader = flc.make_balanced_loader(X_test, y_test)

    clients = []
    client_dist = []
    for sheet in client_sheets:
        cdf = pd.read_excel(xl, sheet)
        X = cdf.iloc[:, :-1].values.astype(np.float32)
        y = cdf.iloc[:, -1].values.astype(np.int64)
        counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        client_dist.append(counts)
        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                             batch_size=BATCH_SIZE, shuffle=True)
        clients.append((loader, len(cdf)))

    input_dim = X_val.shape[1]
    return clients, client_dist, client_sheets, val_loader, test_loader, balanced_test_loader, input_dim


def run_once(model_name, algo, mu, gmin, gmax, scale_factor, seed,
             clients, client_dist, val_loader, test_loader, balanced_test_loader, input_dim,
             counts_df, eligible_clients, quality_scores, probabilities):
    torch.manual_seed(seed)
    np.random.seed(seed)

    class_weights, n0, n1 = flc.compute_class_weights_federated_safe(counts_df, scale_factor=scale_factor)
    class_weights = class_weights.to(flc.device)  # <--- Bổ sung chuyển tensor lên GPU
    criterion = flc.AdaptiveFocalLoss(alpha=class_weights, gamma_min=gmin, gamma_max=gmax)

    global_model = flc.build_model(model_name, input_dim)
    best_val_f1 = -1.0
    best_state = None

    for round_num in range(1, NUM_ROUNDS + 1):
        frac = np.random.uniform(FRACTION_MIN, FRACTION_MAX)
        n_sel = max(1, int(frac * len(eligible_clients)))
        selected = np.random.choice(eligible_clients, size=min(n_sel, len(eligible_clients)),
                                     p=probabilities, replace=False)
        client_models, train_losses = [], []
        for idx in selected:
            local_model = flc.build_model(model_name, input_dim)
            local_model.load_state_dict(global_model.state_dict())
            loader, size = clients[idx]
            if size < BATCH_SIZE:
                continue
            quality = quality_scores[eligible_clients.index(idx)]
            if algo == "fedprox":
                loss = flc.train_local_fedprox(local_model, global_model, loader, criterion,
                                                client_quality=quality, epochs=LOCAL_EPOCHS, proximal_mu=mu)
                if loss in (float("inf"),):
                    continue
                client_models.append((local_model.state_dict(), size, quality))
            else:
                local_model, loss = flc.train_local_fednolowe(local_model, loader, criterion,
                                                                client_quality=quality, epochs=LOCAL_EPOCHS)
                if loss in (float("inf"),):
                    continue
                client_models.append(local_model)
            train_losses.append(loss)

        if not client_models:
            continue

        if algo == "fedprox":
            global_model = flc.aggregate_fedprox(global_model, client_models)
        else:
            global_model = flc.aggregate_nolowe(global_model, client_models, train_losses)

        # Chỉ dùng VALIDATION để chọn round/cấu hình tốt nhất
        val_metrics = flc.evaluate(global_model, val_loader)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    if best_state is not None:
        global_model.load_state_dict(best_state)

    # Đánh giá CUỐI CÙNG trên Test — đúng một lần, không quay lại chọn gì nữa
    test_metrics = flc.evaluate(global_model, test_loader)
    balanced_test_metrics = flc.evaluate(global_model, balanced_test_loader)

    return {
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "balanced_test_macro_f1": balanced_test_metrics["macro_f1"],
        "balanced_test_accuracy": balanced_test_metrics["accuracy"],
        "n0_train": n0, "n1_train": n1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["deepnn", "transformernn", "residualnn"])
    parser.add_argument("--algo", required=True, choices=["fedprox", "fednolowe"])
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--include_low_minority_clients", action="store_true",
                         help="Ablation: bỏ ngưỡng >=8 mẫu minority, cho toàn bộ client tham gia "
                              "(trả lời câu hỏi client-selection bias của R2/R6).")
    args = parser.parse_args()

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Không thấy '{DATA_FILE}'. Hãy chạy prepare_debtcrisisfl.py trước.")

    counts_df = pd.read_csv(COUNTS_FILE)
    (clients, client_dist, client_sheets, val_loader, test_loader,
     balanced_test_loader, input_dim) = load_data()

    if args.include_low_minority_clients:
        eligible_clients = list(range(len(clients)))
    else:
        eligible_clients = [i for i in range(len(clients)) if client_dist[i].get(1, 0) >= MIN_MINORITY_SAMPLES_TO_PARTICIPATE]
    quality_scores = [flc.client_quality_score_enhanced(client_dist[i]) for i in eligible_clients]
    total_q = sum(quality_scores)
    probabilities = [s / total_q for s in quality_scores] if total_q > 0 else [1.0 / len(eligible_clients)] * len(eligible_clients)
    print(f"Client tham gia FL: {len(eligible_clients)} / {len(clients)} tổng số "
          f"(include_low_minority_clients={args.include_low_minority_clients})")

    if FULL_GRID:
        mu_values = [0.1, 0.3, 0.5, 0.7, 1.0] if args.algo == "fedprox" else [0.0]
        gmax_values = [2.0, 3.0, 4.0, 5.0]
        sf_values = [2.0, 3.0, 4.0]
    else:
        mu_values = [0.1, 0.3, 0.5] if args.algo == "fedprox" else [0.0]
        gmax_values = [2.0, 3.0]
        sf_values = [2.0, 3.0]
    gmin_values = [1.0]
    grid = list(itertools.product(mu_values, gmin_values, gmax_values, sf_values))
    print(f"Tổng số cấu hình: {len(grid)} | seeds/cấu hình: {args.n_seeds}")

    out_file = f"gridsearch_{args.model}_{args.algo}.csv"
    fieldnames = ["proximal_mu", "gamma_min", "gamma_max", "scale_factor", "seed",
                  "best_val_macro_f1", "test_macro_f1", "test_accuracy",
                  "test_balanced_accuracy", "balanced_test_macro_f1", "balanced_test_accuracy",
                  "n0_train", "n1_train"]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (mu, gmin, gmax, sf) in grid:
            for seed in range(args.n_seeds):
                result = run_once(args.model, args.algo, mu, gmin, gmax, sf, seed,
                                   clients, client_dist, val_loader, test_loader, balanced_test_loader,
                                   input_dim, counts_df, eligible_clients, quality_scores, probabilities)
                row = {"proximal_mu": mu, "gamma_min": gmin, "gamma_max": gmax, "scale_factor": sf,
                       "seed": seed, **result}
                writer.writerow(row)
                f.flush()
                print(f"[mu={mu}, gmax={gmax}, sf={sf}, seed={seed}] "
                      f"Val F1={result['best_val_macro_f1']:.4f} | "
                      f"Test F1={result['test_macro_f1']:.4f} (natural) | "
                      f"Balanced-Test F1={result['balanced_test_macro_f1']:.4f}")

    print(f"\nĐã ghi kết quả từng seed vào '{out_file}'.")
    print("Gợi ý phân tích: group theo (proximal_mu, gamma_max, scale_factor), "
          "chọn cấu hình có best_val_macro_f1 trung bình cao nhất, rồi báo cáo "
          "mean +/- std của test_macro_f1 / test_balanced_accuracy cho đúng cấu hình đó "
          "(Bảng kết quả chính nên dùng test_macro_f1 trên phân phối tự nhiên; "
          "balanced_test_* chỉ nêu như phân tích phụ).")


if __name__ == "__main__":
    main()
