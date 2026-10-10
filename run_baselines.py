"""
run_baselines.py
=====================
Baseline TẬP TRUNG (centralized) để trả lời yêu cầu của Editor/R1/R2:
"so sánh với baseline không-FL để biết FL có đánh đổi hiệu năng hay không,
và để có căn cứ cho mọi tuyên bố kiểu 'state-of-the-art'".

Ba baseline:
  1. Logistic Regression (class_weight='balanced', không cần nhìn nhãn test)
  2. Random Forest        (class_weight='balanced')
  3. Bản CENTRALIZED của kiến trúc tốt nhất (vd. ResidualNN) — gộp toàn bộ
     dữ liệu Train của tất cả client lại huấn luyện 1 lần (không có vòng FL),
     dùng đúng Global_Val để chọn epoch tốt nhất, Global_Test để báo cáo cuối.

INPUT: data_benchmark.xlsx (từ prepare_debtcrisisfl.py)

Cách chạy:
  python run_baselines.py --centralized_model residualnn --n_seeds 5
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

import fl_common as flc

DATA_FILE = "data_benchmark_MISSING_STRATEGY_mean.xlsx"
BATCH_SIZE = 64
CENTRALIZED_EPOCHS = 30


def load_all():
    xl = pd.ExcelFile(DATA_FILE)
    val_df = pd.read_excel(xl, "Global_Val")
    test_df = pd.read_excel(xl, "Global_Test")
    client_sheets = [s for s in xl.sheet_names if s not in ("Global_Val", "Global_Test")]
    train_parts = [pd.read_excel(xl, s) for s in client_sheets]
    train_df = pd.concat(train_parts, ignore_index=True)

    def split_xy(df):
        return df.iloc[:, :-1].values.astype(np.float32), df.iloc[:, -1].values.astype(np.int64)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)
    return X_train, y_train, X_val, y_val, X_test, y_test


def report(name, y_true, y_pred):
    print(f"[{name}] macro-F1={f1_score(y_true, y_pred, average='macro'):.4f} | "
          f"Acc={accuracy_score(y_true, y_pred):.4f} | "
          f"BalancedAcc={balanced_accuracy_score(y_true, y_pred):.4f}")
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def run_sklearn_baselines(X_train, y_train, X_test, y_test, n_seeds):
    results = []
    for seed in range(n_seeds):
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        lr.fit(X_train, y_train)
        r = report(f"LogReg seed={seed}", y_test, lr.predict(X_test))
        r.update({"model": "LogisticRegression", "seed": seed})
        results.append(r)

        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        r = report(f"RandomForest seed={seed}", y_test, rf.predict(X_test))
        r.update({"model": "RandomForest", "seed": seed})
        results.append(r)
    return pd.DataFrame(results)


def run_centralized_nn(model_name, X_train, y_train, X_val, y_val, X_test, y_test, n_seeds):
    results = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                                   batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                                 batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                                  batch_size=BATCH_SIZE, shuffle=False)

        n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
        counts = np.array([n0, n1], dtype=np.float64)
        w = counts.sum() / (2.0 * counts)  # trọng số nghịch đảo tần suất, học trên TRAIN

        class_weights = torch.tensor(w, dtype=torch.float32).to(flc.device)
        criterion = flc.AdaptiveFocalLoss(alpha=class_weights, gamma_min=1.0, gamma_max=3.0)

        model = flc.build_model(model_name, X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

        best_val_f1, best_state = -1.0, None
        for epoch in range(CENTRALIZED_EPOCHS):
            model.train()
            for data, target in train_loader:
                data, target = data.to(flc.device), target.to(flc.device)
                optimizer.zero_grad()
                loss = criterion(model(data), target.long())
                loss.backward()
                optimizer.step()
            val_metrics = flc.evaluate(model, val_loader)
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_metrics = flc.evaluate(model, test_loader)
        print(f"[Centralized-{model_name} seed={seed}] "
              f"Val F1={best_val_f1:.4f} | Test F1={test_metrics['macro_f1']:.4f}")
        results.append({"model": f"Centralized-{model_name}", "seed": seed,
                         "macro_f1": test_metrics["macro_f1"],
                         "accuracy": test_metrics["accuracy"],
                         "balanced_accuracy": test_metrics["balanced_accuracy"]})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centralized_model", default="residualnn",
                         choices=["deepnn", "transformernn", "residualnn"])
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_all()
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    sk_results = run_sklearn_baselines(X_train, y_train, X_test, y_test, args.n_seeds)
    nn_results = run_centralized_nn(args.centralized_model, X_train, y_train, X_val, y_val,
                                     X_test, y_test, args.n_seeds)

    all_results = pd.concat([sk_results, nn_results], ignore_index=True)
    all_results.to_csv("baselines_results.csv", index=False)

    summary = all_results.groupby("model")[["macro_f1", "accuracy", "balanced_accuracy"]].agg(["mean", "std"])
    print("\n=== Tổng hợp baseline (mean +/- std qua các seed) ===")
    print(summary)
    summary.to_csv("baselines_summary.csv")
    print("\nĐã ghi 'baselines_results.csv' và 'baselines_summary.csv'.")


if __name__ == "__main__":
    main()
