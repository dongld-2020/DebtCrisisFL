"""
prepare_debtcrisisfl.py
===========================
Xây dựng lại benchmark DebtCrisisFL để khắc phục các vấn đề mà Editor và
6 reviewers (PeerJ CS) đã nêu:

  1. Chia dữ liệu theo THỜI GIAN (chronological), không random -> tránh
     temporal leakage cho panel data theo quý.
  2. Chuẩn hoá min-max được HỌC (fit) duy nhất trên tập Train, áp dụng
     lại cho Validation/Test và cho từng client -> tránh mỗi client / tập
     test tự học min-max riêng (rò rỉ nhẹ + không nhất quán thang đo).
  3. Tách MỘT tập Validation (dùng để chọn hyperparameter, early-stop,
     chọn round tốt nhất) và MỘT tập Test độc lập (chỉ đánh giá 1 lần,
     không dùng để chọn gì cả) -> đúng yêu cầu editor.
  4. Cho phép loại bỏ đặc trưng có khả năng "circular" với nhãn:
     CrisisIndex3(TotalDebt%) được xác nhận bằng thực nghiệm là một
     ngưỡng cứng trên chính đặc trưng TotalDebt% (~77%). Vì vậy mặc định
     script loại bỏ TotalDebt% khỏi tập đặc trưng đầu vào khi nhãn dùng
     là CrisisIndex (xem EXCLUDE_CIRCULAR_FEATURES bên dưới). Có thể tắt
     để so sánh ablation.
  5. Cho phép chọn horizon dự báo H quý (H=0: nowcasting/phân loại cùng
     kỳ như bản gốc; H>0: dự báo trước H quý, dùng đặc trưng tại t để dự
     báo nhãn tại t+H -> trả lời yêu cầu "forecast horizon" của R3/R5/R6).
  6. Tuỳ chọn chiến lược impute: 'zero' (như bài gốc) hoặc 'mean'
     (học trên Train), có thể bật thêm cờ missing-indicator theo đặc trưng.

INPUT BẮT BUỘC
---------------
File: Raw_Data_Final_240723.xlsx
Sheet: 'Data-raw'  (có cột Year, Quarter, QuarterCode, 26 đặc trưng gốc,
                    CrisisIndex1..4, CrisisIndexTotal (0-4, rời rạc),
                    CrisisIndex (nhị phân 0/1 -- ĐÂY LÀ NHÃN đã dùng
                    trong bản thảo gốc, bị đặt tên là "CrisisIndexTotal"
                    trong bài báo; xem ghi chú trong rebuttal letter).

Đặt file này cùng thư mục với script, hoặc sửa INPUT_FILE bên dưới.

OUTPUT
------
data_benchmark.xlsx với các sheet:
  - Global_Val   : tập validation (chọn hyperparameter)
  - Global_Test  : tập test (chỉ đánh giá cuối cùng, dùng đúng 1 lần)
  - <COUNTRY_CODE>: một sheet / client, CHỈ chứa dữ liệu trong giai đoạn Train

Ngoài ra ghi thêm client_train_class_counts.csv: số lượng mẫu Class 0/1 theo
từng client trong tập Train -- dùng để tính class weight kiểu "federated"
(server chỉ nhận số đếm tổng hợp từ client, KHÔNG cần nhìn nhãn test) thay
vì compute_class_weights(y_test, ...) như bản gốc.
"""

import os
import re
import json
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CẤU HÌNH -- chỉnh ở đây, không cần sửa phần code bên dưới
# ----------------------------------------------------------------------
INPUT_FILE = "Raw_Data_Final_240723.xlsx"
INPUT_SHEET = "Data-raw"
OUTPUT_FILE = "data_benchmark_MISSING_STRATEGY_mean.xlsx"
COUNTS_FILE = "client_train_class_counts.csv"

LABEL_COL = "CrisisIndex"          # nhãn nhị phân đã xác nhận khớp bài báo
CODE_COL = "Code"
YEAR_COL = "Year"
QUARTER_COL = "Quarter"            # dạng 'Q1'..'Q4'
QUARTERCODE_COL = "QuarterCode"    # dạng 'ABW2000Q1', dùng để sort theo thời gian

# Mốc chia theo năm (chronological). Có thể chỉnh lại tuỳ ý.
# Train: năm <= TRAIN_END_YEAR
# Val  : TRAIN_END_YEAR < năm <= VAL_END_YEAR
# Test : năm > VAL_END_YEAR
TRAIN_END_YEAR = 2015
VAL_END_YEAR = 2018

# Horizon dự báo (đơn vị: quý). 0 = nowcasting (giống bản gốc, nhưng nay
# đã leakage-free nhờ chia theo thời gian). >0 = dùng đặc trưng tại t để
# dự báo CrisisIndex tại t+H trong CÙNG một quốc gia.
FORECAST_HORIZON = 0

# Loại bỏ đặc trưng bị xác nhận là circular với nhãn (ngưỡng cứng
# TotalDebt% ~ 77% quyết định CrisisIndex3, một trong 4 thành phần cấu
# thành CrisisIndex). Đặt False để chạy ablation "giữ nguyên" và so sánh.
EXCLUDE_CIRCULAR_FEATURES = False
CIRCULAR_FEATURES = ["TotalDebt%"]

# 'zero' (giống bài gốc) hoặc 'mean' (học trên Train, áp cho Val/Test/client)
MISSING_STRATEGY = "mean"
ADD_MISSING_INDICATOR = False   # nếu True, thêm cột nhị phân "<feat>__was_missing"

# Các cột KHÔNG phải đặc trưng kinh tế (id, meta, nhãn, thành phần nhãn,
# và các cột "cơ chế" khoản vay IMF chỉ dùng để dựng CrisisIndex1/2, không
# phải đặc trưng dự báo trong bài báo gốc).
NON_FEATURE_COLS = [
    "Code", "Quarter", "Year", "QuarterCode", "YrCode",
    "Member", "Member Code", "mnthcode", "yrcode2", "Type", "Facility",
    "Date Of Commitment", "month", "year3", "Expiration Date",
    "commitmentLoan", "% Of Quota", "Precautionary", "NAB Eligible",
    "Expiration Date2", "accumulatedLoans", "% Of Quota4", "Amount Drawn",
    "Amount Outstanding", "Status", "GDP (current US$)", "GDP monthly",
    "GDP (current US$) [NY.MKTP.CD]",
    "CrisisIndex1(100%quota)", "CrisisIndex2(200%quota)",
    "CrisisIndex3(TotalDebt%)", "CrisisIndex4(HighRisk)",
    "CrisisIndexTotal", "CrisisIndex",
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_raw(input_file, sheet):
    df = pd.read_excel(input_file, sheet_name=sheet)
    missing_needed = [c for c in [CODE_COL, YEAR_COL, LABEL_COL] if c not in df.columns]
    if missing_needed:
        raise ValueError(f"Thiếu cột bắt buộc trong sheet '{sheet}': {missing_needed}")
    return df


def build_feature_list(df):
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if EXCLUDE_CIRCULAR_FEATURES:
        feats = [c for c in feats if c not in CIRCULAR_FEATURES]
    return feats


def apply_forecast_horizon(df, horizon):
    """Dịch nhãn về trước H quý trong cùng một quốc gia: đặc trưng tại t
    được gán với nhãn tại t+H. Yêu cầu QuarterCode sort được theo thời gian.
    """
    if horizon <= 0:
        return df
    df = df.sort_values([CODE_COL, YEAR_COL, QUARTER_COL]).reset_index(drop=True)
    df["_label_shifted"] = df.groupby(CODE_COL)[LABEL_COL].shift(-horizon)
    df = df.dropna(subset=["_label_shifted"]).copy()
    df[LABEL_COL] = df["_label_shifted"].astype(int)
    df = df.drop(columns=["_label_shifted"])
    print(f"[horizon={horizon}] Sau khi dịch nhãn: {len(df)} dòng còn lại "
          f"(mỗi quốc gia mất {horizon} quý cuối chuỗi thời gian).")
    return df


def chronological_split(df):
    train_mask = df[YEAR_COL] <= TRAIN_END_YEAR
    val_mask = (df[YEAR_COL] > TRAIN_END_YEAR) & (df[YEAR_COL] <= VAL_END_YEAR)
    test_mask = df[YEAR_COL] > VAL_END_YEAR
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    print(f"Train: {len(train_df)} mẫu (<= {TRAIN_END_YEAR}) | "
          f"Val: {len(val_df)} mẫu ({TRAIN_END_YEAR+1}-{VAL_END_YEAR}) | "
          f"Test: {len(test_df)} mẫu (> {VAL_END_YEAR})")
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        vc = part[LABEL_COL].value_counts(normalize=True) * 100
        print(f"  {name} class balance: {vc.to_dict()}")
    return train_df, val_df, test_df


def fit_impute_and_scale_stats(train_df, feature_cols):
    """Học min/max (và mean nếu cần impute='mean') CHỈ trên Train."""
    stats = {}
    for col in feature_cols:
        col_train = train_df[col]
        if MISSING_STRATEGY == "mean":
            fill_value = col_train.mean()
            if pd.isna(fill_value):
                fill_value = 0.0
        else:
            fill_value = 0.0
        filled = col_train.fillna(fill_value)
        mn, mx = filled.min(), filled.max()
        rng = (mx - mn) if (mx - mn) != 0 else 1.0
        stats[col] = {"fill": float(fill_value), "min": float(mn), "max": float(mx), "range": float(rng)}
    return stats


def apply_impute_and_scale(df, feature_cols, stats):
    df = df.copy()
    for col in feature_cols:
        s = stats[col]
        if ADD_MISSING_INDICATOR:
            df[f"{col}__was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(s["fill"])
        df[col] = (df[col] - s["min"]) / s["range"]
    return df


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Không tìm thấy '{INPUT_FILE}'. Hãy đặt file Raw_Data_Final_240723.xlsx "
            f"cùng thư mục với script này, hoặc sửa biến INPUT_FILE."
        )

    df = load_raw(INPUT_FILE, INPUT_SHEET)
    print(f"Đã đọc {len(df)} dòng từ sheet '{INPUT_SHEET}'.")

    feature_cols = build_feature_list(df)
    print(f"Số đặc trưng sử dụng: {len(feature_cols)} "
          f"(đã {'LOẠI BỎ' if EXCLUDE_CIRCULAR_FEATURES else 'GIỮ'} "
          f"{CIRCULAR_FEATURES} do nghi ngờ circular với nhãn).")

    # --- BỔ SUNG: Ép kiểu toàn bộ đặc trưng về dạng số ---
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # ----------------------------------------------------    
    
    # Chỉ giữ các cột cần thiết + parse Quarter số để sort/shift chuẩn
    keep_cols = [CODE_COL, YEAR_COL, QUARTER_COL, QUARTERCODE_COL, LABEL_COL] + feature_cols
    df = df[keep_cols].copy()
    df[QUARTER_COL] = df[QUARTER_COL].astype(str).str.extract(r"(\d)").astype(int)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    df = apply_forecast_horizon(df, FORECAST_HORIZON)

    train_df, val_df, test_df = chronological_split(df)

    stats = fit_impute_and_scale_stats(train_df, feature_cols)
    train_df = apply_impute_and_scale(train_df, feature_cols, stats)
    val_df = apply_impute_and_scale(val_df, feature_cols, stats)
    test_df = apply_impute_and_scale(test_df, feature_cols, stats)

    out_feature_cols = feature_cols + ([f"{c}__was_missing" for c in feature_cols] if ADD_MISSING_INDICATOR else [])

    global_val = val_df[out_feature_cols + [LABEL_COL]].rename(columns={LABEL_COL: "CrisisIndexTotal"})
    global_test = test_df[out_feature_cols + [LABEL_COL]].rename(columns={LABEL_COL: "CrisisIndexTotal"})

    # Phân client theo Code, CHỈ trong giai đoạn Train
    client_groups = train_df.groupby(CODE_COL)
    counts_rows = []
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        global_val.to_excel(writer, sheet_name="Global_Val", index=False)
        global_test.to_excel(writer, sheet_name="Global_Test", index=False)
        for client_id, cdf in client_groups:
            out = cdf[out_feature_cols + [LABEL_COL]].rename(columns={LABEL_COL: "CrisisIndexTotal"})
            sheet_name = str(client_id)[:31]  # giới hạn tên sheet Excel
            out.to_excel(writer, sheet_name=sheet_name, index=False)
            vc = cdf[LABEL_COL].value_counts()
            counts_rows.append({
                "client_id": client_id,
                "n_class0": int(vc.get(0, 0)),
                "n_class1": int(vc.get(1, 0)),
                "n_total": int(len(cdf)),
            })

    pd.DataFrame(counts_rows).to_csv(COUNTS_FILE, index=False)

    meta = {
        "input_file": INPUT_FILE,
        "input_sheet": INPUT_SHEET,
        "label_col_used": LABEL_COL,
        "train_end_year": TRAIN_END_YEAR,
        "val_end_year": VAL_END_YEAR,
        "forecast_horizon_quarters": FORECAST_HORIZON,
        "exclude_circular_features": EXCLUDE_CIRCULAR_FEATURES,
        "circular_features_excluded": CIRCULAR_FEATURES if EXCLUDE_CIRCULAR_FEATURES else [],
        "missing_strategy": MISSING_STRATEGY,
        "add_missing_indicator": ADD_MISSING_INDICATOR,
        "n_features": len(out_feature_cols),
        "n_clients": len(counts_rows),
        "n_val": len(global_val),
        "n_test": len(global_test),
    }
    with open("benchmark_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nHoàn tất. Đã ghi '{OUTPUT_FILE}', '{COUNTS_FILE}', 'benchmark_meta.json'.")
    print(f"Số client: {len(counts_rows)} | Val: {len(global_val)} mẫu | Test: {len(global_test)} mẫu")


if __name__ == "__main__":
    main()
