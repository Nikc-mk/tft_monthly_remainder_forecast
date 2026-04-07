import numpy as np
import pandas as pd
import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mase
from darts.dataprocessing.transformers import (
    StaticCovariatesTransformer,
    WindowTransformer,
    Scaler,
)
from darts.utils.missing_values import fill_missing_values

ru_holidays = holidays.RU()


def finalize_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()
    else:
        print(f"Plot saved to {filename}")
    plt.close()

# =========================
# 1) Загрузка и подготовка
# =========================
df = pd.read_csv("data/sales.csv", sep=";", parse_dates=["Week"])

cities = sorted(df["City"].unique())

ts_target_list = TimeSeries.from_group_dataframe(
    df,
    group_cols="City",
    time_col="Week",
    value_cols="revenue",
    fill_missing_dates=True,
    freq="W-MON",
    fillna_value=0.0
)

# City -> numeric static covariates
static_transformer = StaticCovariatesTransformer()
ts_target_list = static_transformer.fit_transform(ts_target_list)

# =========================
# 2) Rolling / EWM признаки
# =========================
rolling_transformer = WindowTransformer(
    transforms=[
        {"function": "mean",   "mode": "rolling", "window": 4, "function_name": "mean_4"},
        {"function": "mean",   "mode": "rolling", "window": 8, "function_name": "mean_8"},
        {"function": "std",    "mode": "rolling", "window": 4, "function_name": "std_4"},
        {"function": "min",    "mode": "rolling", "window": 4, "function_name": "min_4"},
        {"function": "max",    "mode": "rolling", "window": 4, "function_name": "max_4"},
        {"function": "median", "mode": "rolling", "window": 4, "function_name": "median_4"},
        {"function": "mean",   "mode": "ewm",     "span": 4,   "function_name": "ewm_4"},
    ],
    include_current=False,
    forecasting_safe=True,
    treat_na="dropna",
)

rolling_covs = rolling_transformer.transform(ts_target_list)
rolling_covs = [fill_missing_values(ts, fill=0.0) for ts in rolling_covs]

# выравниваем target и covariates
target_aligned = []
past_covs_aligned = []

for ts, rc in zip(ts_target_list, rolling_covs):
    common_start = max(ts.start_time(), rc.start_time())
    common_end = min(ts.end_time(), rc.end_time())

    target_aligned.append(ts.slice(common_start, common_end))
    past_covs_aligned.append(rc.slice(common_start, common_end))

# =========================
# 3) Календарные future features
# =========================
def absolute_week_idx(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        ((idx - idx.min()).days // 7).astype("int64"),
        index=idx,
        name="abs_week_idx"
    )


def ru_holiday_days_in_week(idx: pd.DatetimeIndex) -> pd.Series:
    vals = []
    for dt in idx:
        week_start = dt - pd.Timedelta(days=dt.weekday())
        week_days = pd.date_range(week_start, periods=7, freq="D")
        cnt = sum(
            (d.weekday() >= 5) or (d.date() in ru_holidays)
            for d in week_days
        )
        vals.append(cnt)

    return pd.Series(vals, index=idx, name="ru_holiday_days_in_week", dtype=np.float32)

# =========================
# 4) Split: train / val / blind test
# =========================
VAL_HORIZON = 8
TEST_HORIZON = 8
MASE_M = 1   # для недельных данных базовый вариант; можно потом попробовать 52

train_series = []
val_series = []
test_series = []

train_past_covs = []
val_past_covs_full = []
test_past_covs_full = []

for ts, pc in zip(target_aligned, past_covs_aligned):
    train_end = len(ts) - VAL_HORIZON - TEST_HORIZON
    val_end = len(ts) - TEST_HORIZON

    ts_train = ts[:train_end]
    ts_val = ts[train_end:val_end]
    ts_test = ts[val_end:]

    pc_train = pc[:train_end]
    pc_val_full = pc[:val_end]
    pc_test_full = pc

    train_series.append(ts_train)
    val_series.append(ts_val)
    test_series.append(ts_test)

    train_past_covs.append(pc_train)
    val_past_covs_full.append(pc_val_full)
    test_past_covs_full.append(pc_test_full)

# =========================
# 5) Scaling
# =========================
target_scaler = Scaler()
past_cov_scaler = Scaler()

train_series_scaled = target_scaler.fit_transform(train_series)
val_series_scaled = target_scaler.transform(val_series)
test_series_scaled = target_scaler.transform(test_series)

train_past_covs_scaled = past_cov_scaler.fit_transform(train_past_covs)
val_past_covs_full_scaled = past_cov_scaler.transform(val_past_covs_full)
test_past_covs_full_scaled = past_cov_scaler.transform(test_past_covs_full)

# =========================
# 6) TFT model
# =========================
MODEL_KWARGS = dict(
    input_chunk_length=24,
    output_chunk_length=2,
    hidden_size=16,
    lstm_layers=1,
    num_attention_heads=2,
    dropout=0.1,
    batch_size=32,
    n_epochs=10,
    random_state=42,
    add_encoders={
        "datetime_attribute": {"future": ["year", "month", "quarter", "week"]},
        "cyclic": {"future": ["month"]},
        "custom": {"future": [absolute_week_idx, ru_holiday_days_in_week]},
    },
)

model = TFTModel(**MODEL_KWARGS)

# =========================
# 7) Fit on TRAIN only
# =========================
model.fit(
    series=train_series_scaled,
    past_covariates=train_past_covs_scaled,
    verbose=True,
)

# =========================
# 8) Validation forecast
# =========================
val_preds_scaled = model.predict(
    n=VAL_HORIZON,
    series=train_series_scaled,
    past_covariates=val_past_covs_full_scaled,
)

val_preds = target_scaler.inverse_transform(val_preds_scaled)

print("Validation MASE by city:")
val_mases = []
for city, insample, actual, pred in zip(cities, train_series, val_series, val_preds):
    score = mase(actual, pred, insample=insample, m=MASE_M)
    val_mases.append(score)
    print(f"{city}: {score:.3f}")

# общий MASE по всем сериям сразу
val_mase_global = mase(
    val_series,
    val_preds,
    insample=train_series,
    m=MASE_M,
    series_reduction=np.nanmean,
)
print(f"\nGlobal validation MASE: {val_mase_global:.3f}")

# =========================
# 9) Retrain on TRAIN+VAL, then blind TEST forecast
# =========================
trainval_series = []
trainval_past_covs = []

for ts, pc in zip(target_aligned, past_covs_aligned):
    trainval_end = len(ts) - TEST_HORIZON
    trainval_series.append(ts[:trainval_end])
    trainval_past_covs.append(pc[:trainval_end])

target_scaler_tv = Scaler()
past_cov_scaler_tv = Scaler()

trainval_series_scaled = target_scaler_tv.fit_transform(trainval_series)
test_series_scaled_2 = target_scaler_tv.transform(test_series)

trainval_past_covs_scaled = past_cov_scaler_tv.fit_transform(trainval_past_covs)
test_past_covs_full_scaled_2 = past_cov_scaler_tv.transform(test_past_covs_full)

model_test = TFTModel(**MODEL_KWARGS)

model_test.fit(
    series=trainval_series_scaled,
    past_covariates=trainval_past_covs_scaled,
    verbose=True,
)

test_preds_scaled = model_test.predict(
    n=TEST_HORIZON,
    series=trainval_series_scaled,
    past_covariates=test_past_covs_full_scaled_2,
)

test_preds = target_scaler_tv.inverse_transform(test_preds_scaled)

print("\nBlind TEST MASE by city:")
test_mases = []
for city, insample, actual, pred in zip(cities, trainval_series, test_series, test_preds):
    score = mase(actual, pred, insample=insample, m=MASE_M)
    test_mases.append(score)
    print(f"{city}: {score:.3f}")

# общий MASE по всем тестовым сериям
test_mase_global = mase(
    test_series,
    test_preds,
    insample=trainval_series,
    m=MASE_M,
    series_reduction=np.nanmean,
)
print(f"\nGlobal blind TEST MASE: {test_mase_global:.3f}")

# =========================
# 10) График MASE по тесту
# =========================
mase_df = pd.DataFrame({
    "City": cities,
    "MASE": test_mases,
}).sort_values("MASE")

plt.figure(figsize=(10, 5))
plt.bar(mase_df["City"], mase_df["MASE"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("MASE")
plt.title(f"Blind TEST MASE by City | Global MASE = {test_mase_global:.3f}")
finalize_plot("test_mase_by_city.png")

# =========================
# 11) График test forecast по неделям
# =========================
city_idx = 0

plt.figure(figsize=(12, 5))
test_series[city_idx].plot(label="actual test")
test_preds[city_idx].plot(label="predicted test")

city_mase = mase(
    test_series[city_idx],
    test_preds[city_idx],
    insample=trainval_series[city_idx],
    m=MASE_M
)

plt.title(
    f"Blind TEST Forecast — {cities[city_idx]} | "
    f"MASE = {city_mase:.3f}"
)
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.legend()
finalize_plot("test_forecast_by_week.png")

# =========================
# 12) График weekly MAPE по test forecast
# =========================
actual_test_values = test_series[city_idx].values(copy=False).reshape(-1)
pred_test_values = test_preds[city_idx].values(copy=False).reshape(-1)
test_weekly_mape = np.where(
    np.abs(actual_test_values) > 1e-8,
    np.abs((actual_test_values - pred_test_values) / actual_test_values) * 100.0,
    np.nan,
)

plt.figure(figsize=(12, 5))
plt.plot(
    test_series[city_idx].time_index,
    test_weekly_mape,
    marker="o",
    label="weekly MAPE",
)
plt.title(
    f"Blind TEST Weekly MAPE - {cities[city_idx]} | "
    f"Mean = {np.nanmean(test_weekly_mape):.2f}%"
)
plt.xlabel("Week")
plt.ylabel("MAPE, %")
plt.xticks(rotation=45)
plt.legend()
finalize_plot("test_mape_by_week.png")
