import os
os.environ["XGBOOST_DISABLE_VMM"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =========================================================
# Config
# =========================================================
DATA_DIR = Path("/data")
TARGET = "avg_delay_minutes_next_30m"
ID_COL = "ID"
N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS_XGB = 80
N_TRIALS_LGB = 80
EPS = 1e-6
N_LAYOUT_CLUSTERS = 20


# =========================================================
# Target transforms
# =========================================================
def t_sqrt(y):   return np.sqrt(np.clip(np.asarray(y, dtype=float), 0, None))
def inv_sqrt(y): return np.clip(np.asarray(y, dtype=float), 0, None) ** 2


# =========================================================
# Load data
# =========================================================
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
layout_info = pd.read_csv(DATA_DIR / "layout_info.csv")
sample_sub  = pd.read_csv(DATA_DIR / "sample_submission.csv")

train = train.merge(layout_info, on="layout_id", how="left")
test  = test.merge(layout_info, on="layout_id", how="left")
print(f"train {train.shape}, test {test.shape}")


# =========================================================
# v16 NEW: Layout clustering
# layout_info는 300개 레이아웃 전부 포함 (test-only 50개 포함)
# → unseen layout도 cluster TE 혜택을 받을 수 있음
# =========================================================
def make_layout_clusters(layout_info, n_clusters=N_LAYOUT_CLUSTERS):
    num_cols = [
        "aisle_width_avg", "intersection_count", "one_way_ratio",
        "pack_station_count", "charger_count", "layout_compactness",
        "zone_dispersion", "robot_total", "building_age_years",
        "floor_area_sqm", "ceiling_height_m", "fire_sprinkler_count",
        "emergency_exit_count",
    ]
    num_cols = [c for c in num_cols if c in layout_info.columns]
    X = layout_info[num_cols].fillna(layout_info[num_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    layout_info = layout_info.copy()
    layout_info["layout_cluster"] = km.fit_predict(X_scaled).astype("int16")
    print(f"Layout clusters: {n_clusters} clusters for {len(layout_info)} layouts")
    return layout_info[["layout_id", "layout_cluster"]]

layout_clusters = make_layout_clusters(layout_info)
train = train.merge(layout_clusters, on="layout_id", how="left")
test  = test.merge(layout_clusters, on="layout_id", how="left")


# =========================================================
# v16 NEW: Scenario-level context features
# v17 UPDATE: SCENE_COLS 확장 (5→12)
# v18 UPDATE: SCENE_COLS 확장 (12→18)
# =========================================================
SCENE_COLS = [
    "charging_ratio_raw",
    "congestion_score",
    "low_battery_ratio",
    "order_inflow_15m",
    "robot_utilization",
    # v17 NEW
    "fault_count_15m",
    "near_collision_15m",
    "blocked_path_15m",
    "avg_trip_distance",
    "max_zone_density",
    "task_reassign_15m",
    "avg_recovery_time",
    # v18 NEW
    "battery_mean",
    "avg_charge_wait",
    "charge_efficiency_pct",
    "aisle_traffic_score",
    "path_optimization_score",
    "intersection_wait_time_avg",
]

def precompute_scene_ratios(df):
    df = df.copy()
    if "robot_charging" in df.columns and "robot_total" in df.columns:
        df["charging_ratio_raw"] = df["robot_charging"] / (df["robot_total"] + EPS)
    return df

train = precompute_scene_ratios(train)
test  = precompute_scene_ratios(test)

def make_scenario_context(train_df, test_df):
    scene_cols = [c for c in SCENE_COLS if c in train_df.columns]

    def _agg(df, sid_col="scenario_id"):
        result = df.copy()
        for col in scene_cols:
            stats = df.groupby(sid_col)[col].agg(["mean", "std", "max", "min"])
            stats.columns = [
                f"scene_{col}_mean", f"scene_{col}_std",
                f"scene_{col}_max",  f"scene_{col}_min",
            ]
            stats["scene_{}_range".format(col)] = stats[f"scene_{col}_max"] - stats[f"scene_{col}_min"]
            stats = stats.reset_index()
            result = result.merge(stats, on=sid_col, how="left")
        return result

    train_df = _agg(train_df)
    test_df  = _agg(test_df)
    n_new = len([c for c in SCENE_COLS if c in train_df.columns]) * 5
    print(f"Scenario context features added: ~{n_new}")
    return train_df, test_df

train, test = make_scenario_context(train, test)


# =========================================================
# v17 NEW: Timeslot rank within scenario
# 시나리오 내 25개 슬롯 중 몇 번째인지 (0~24)
# =========================================================
def add_timeslot_rank(df):
    df = df.copy()
    df["timeslot_rank"] = df.groupby("scenario_id").cumcount().astype("int8")
    df["timeslot_rank_norm"] = (df["timeslot_rank"] / 24.0).astype("float32")
    df["timeslot_sin"] = np.sin(2 * np.pi * df["timeslot_rank"] / 25).astype("float32")
    df["timeslot_cos"] = np.cos(2 * np.pi * df["timeslot_rank"] / 25).astype("float32")
    return df

train = add_timeslot_rank(train)
test  = add_timeslot_rank(test)
print("Timeslot rank features added")


# =========================================================
# v18 NEW: Lag & diff features within scenario
# 이전 timeslot 값으로 추세(악화/개선)를 피처화
# =========================================================
LAG_COLS = [
    "congestion_score", "charging_ratio_raw", "low_battery_ratio",
    "order_inflow_15m", "blocked_path_15m",   "near_collision_15m",
]
LAG_SCENE_FILL = {
    "congestion_score":    "scene_congestion_score_mean",
    "charging_ratio_raw":  "scene_charging_ratio_raw_mean",
    "low_battery_ratio":   "scene_low_battery_ratio_mean",
    "order_inflow_15m":    "scene_order_inflow_15m_mean",
    "blocked_path_15m":    "scene_blocked_path_15m_mean",
    "near_collision_15m":  "scene_near_collision_15m_mean",
}

def add_lag_features(df):
    orig_index = df.index  # 원래 행 순서 저장
    df = df.sort_values(["scenario_id", "timeslot_rank"]).copy()
    for col in LAG_COLS:
        if col not in df.columns:
            continue
        g = df.groupby("scenario_id")[col]
        df[f"{col}_lag1"]  = g.shift(1)
        df[f"{col}_lag2"]  = g.shift(2)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        # 첫 슬롯 NaN → scene mean으로 대체
        fill_col = LAG_SCENE_FILL.get(col)
        if fill_col and fill_col in df.columns:
            df[f"{col}_lag1"]  = df[f"{col}_lag1"].fillna(df[fill_col])
            df[f"{col}_lag2"]  = df[f"{col}_lag2"].fillna(df[fill_col])
        df[f"{col}_diff1"] = df[f"{col}_diff1"].fillna(0)
    n_new = sum(1 for c in LAG_COLS if c in df.columns) * 3
    print(f"Lag/diff features added: ~{n_new}")
    return df.reindex(orig_index)  # 원래 행 순서 복원

train = add_lag_features(train)
test  = add_lag_features(test)


# =========================================================
# Feature engineering (v15 base + v16 + v17 additions)
# =========================================================
def make_features(df):
    df = df.copy()

    # NaN indicators for operationally critical columns
    for c in ["congestion_score", "fault_count_15m", "near_collision_15m",
              "avg_recovery_time", "charge_queue_length", "avg_charge_wait",
              "battery_std", "robot_utilization", "blocked_path_15m"]:
        if c in df.columns:
            df[f"{c}_nan"] = df[c].isna().astype("int8")

    def _r(n, d):
        if n in df.columns and d in df.columns:
            return df[n] / (df[d] + EPS)
        return None

    def _p(a, b):
        if a in df.columns and b in df.columns:
            return df[a] * df[b]
        return None

    def _s(col, val):
        if val is not None:
            df[col] = val

    # ----- Robot state ratios -----
    _s("idle_ratio",      _r("robot_idle",     "robot_total"))
    _s("charging_ratio",  _r("robot_charging", "robot_total"))
    _s("active_ratio",    _r("robot_active",   "robot_total"))
    _s("utilization_gap", _r("robot_idle",     "robot_active"))

    # charging_ratio polynomial (dominant feature)
    if "charging_ratio" in df.columns:
        cr = df["charging_ratio"]
        df["charging_ratio_sq"]    = cr ** 2
        df["charging_ratio_cube"]  = cr ** 3
        df["charging_ratio_log1p"] = np.log1p(cr.fillna(0))

    # Available robots (active + idle)
    if "robot_active" in df.columns and "robot_idle" in df.columns:
        df["robot_available"] = df["robot_active"] + df["robot_idle"]
    _s("availability_ratio", _r("robot_available", "robot_total"))

    # v16: availability_ratio polynomial (top feature at 30.7%, was missing poly)
    if "availability_ratio" in df.columns:
        ar = df["availability_ratio"]
        df["availability_ratio_sq"]    = ar ** 2
        df["availability_ratio_log1p"] = np.log1p(ar.fillna(0).clip(0))
        df["availability_ratio_inv"]   = 1.0 / (ar.fillna(1.0) + EPS)

    # v17: availability_ratio interactions
    _s("avail_x_congestion",     _p("availability_ratio", "congestion_score"))
    _s("avail_x_order_pressure", _p("availability_ratio", "order_per_active_robot"))
    _s("avail_x_low_battery",    _p("availability_ratio", "low_battery_ratio"))

    # ----- Congestion & density -----
    for col, a, b in [
        ("congestion_x_density",       "congestion_score",   "max_zone_density"),
        ("collision_x_density",        "near_collision_15m", "max_zone_density"),
        ("blocked_x_congestion",       "blocked_path_15m",   "congestion_score"),
        ("blocked_x_collision",        "blocked_path_15m",   "near_collision_15m"),
        ("fault_x_congestion",         "fault_count_15m",    "congestion_score"),
        ("fault_x_density",            "fault_count_15m",    "max_zone_density"),
        ("intersection_x_congestion",  "intersection_count", "congestion_score"),
    ]:
        _s(col, _p(a, b))
    _s("congestion_per_intersection", _r("congestion_score", "intersection_count"))

    # ----- Order throughput -----
    _s("order_per_pack_station", _r("order_inflow_15m", "pack_station_count"))
    _s("order_per_robot",        _r("order_inflow_15m", "robot_total"))
    _s("order_per_active_robot", _r("order_inflow_15m", "robot_active"))
    _s("sku_per_order",          _r("unique_sku_15m",   "order_inflow_15m"))
    _s("sku_concentration",      _r("order_inflow_15m", "unique_sku_15m"))

    if "order_inflow_15m" in df.columns and "pack_station_count" in df.columns:
        df["pack_utilization"] = df["order_inflow_15m"] / (df["pack_station_count"] + EPS)
    _s("pack_util_x_order",      _p("pack_utilization", "order_inflow_15m"))
    _s("pack_util_x_congestion", _p("pack_utilization", "congestion_score"))

    # ----- Battery & charging -----
    _s("battery_risk_score",         _p("low_battery_ratio", "robot_idle"))
    _s("charging_pressure",          _p("low_battery_ratio", "robot_charging"))
    _s("battery_x_congestion",       _p("low_battery_ratio", "congestion_score"))
    _s("charging_robot_per_charger", _r("robot_charging", "charger_count"))
    _s("queue_pressure",             _p("charge_queue_length", "charging_ratio"))

    # charging_ratio × other key features
    if "charging_ratio" in df.columns:
        for col, b in [
            ("charging_x_pack_util",      "pack_utilization"),
            ("charging_x_congestion",     "congestion_score"),
            ("charging_x_order_pressure", "order_per_active_robot"),
            ("charging_x_blocked",        "blocked_path_15m"),
            ("charging_x_battery",        "low_battery_ratio"),
        ]:
            _s(col, _p("charging_ratio", b))

    # ----- Layout & movement -----
    _s("trip_x_recovery",      _p("avg_trip_distance",  "avg_recovery_time"))
    _s("layout_x_density",     _p("layout_compactness", "max_zone_density"))
    _s("dispersion_x_density", _p("zone_dispersion",    "max_zone_density"))

    if "charging_ratio" in df.columns and "layout_x_density" in df.columns:
        df["charging_x_layout_density"] = df["charging_ratio"] * df["layout_x_density"]

    # v17: avg_trip_distance interactions (LGB 2위 피처)
    _s("trip_x_congestion", _p("avg_trip_distance", "congestion_score"))
    _s("trip_x_density",    _p("avg_trip_distance", "max_zone_density"))
    _s("trip_x_order",      _p("avg_trip_distance", "order_per_active_robot"))

    # v17: 미사용 유망 컬럼 interactions
    _s("path_opt_x_congestion",     _p("path_optimization_score",  "congestion_score"))
    _s("intersection_wait_x_count", _p("intersection_wait_time_avg", "intersection_count"))
    _s("charge_eff_x_charging",     _p("charge_efficiency_pct",    "charging_ratio"))
    _s("wms_x_order",               _p("wms_response_time_ms",     "order_inflow_15m"))
    _s("agv_success_x_util",        _p("agv_task_success_rate",    "robot_utilization"))

    # ----- Difference / sum -----
    if "robot_idle" in df.columns and "robot_active" in df.columns:
        df["idle_minus_active"] = df["robot_idle"] - df["robot_active"]
    if "robot_idle" in df.columns and "robot_charging" in df.columns:
        df["idle_plus_charging"] = df["robot_idle"] + df["robot_charging"]

    # ----- Composite risk proxy -----
    req = ["order_per_active_robot", "congestion_score", "low_battery_ratio"]
    if all(c in df.columns for c in req):
        df["delay_risk_proxy"] = (
            df["order_per_active_robot"]
            * (1 + df["congestion_score"])
            * (1 + df["low_battery_ratio"])
        )
        if "charging_ratio" in df.columns:
            df["charging_x_delay_risk"] = df["charging_ratio"] * df["delay_risk_proxy"]

    # ----- Shift hour cyclical encoding -----
    if "shift_hour" in df.columns:
        df["shift_hour_sin"] = np.sin(2 * np.pi * df["shift_hour"] / 24)
        df["shift_hour_cos"] = np.cos(2 * np.pi * df["shift_hour"] / 24)
        if "charging_ratio" in df.columns:
            df["charging_x_shift_sin"] = df["charging_ratio"] * df["shift_hour_sin"]

    # v16: scenario context × current state interactions
    # "현재 상태가 시나리오 평균 대비 얼마나 나쁜가?"
    if "charging_ratio" in df.columns and "scene_charging_ratio_raw_mean" in df.columns:
        df["charging_vs_scene_mean"]  = df["charging_ratio"] - df["scene_charging_ratio_raw_mean"]
        df["charging_vs_scene_ratio"] = df["charging_ratio"] / (df["scene_charging_ratio_raw_mean"] + EPS)

    if "congestion_score" in df.columns and "scene_congestion_score_mean" in df.columns:
        df["congestion_vs_scene_mean"] = df["congestion_score"] - df["scene_congestion_score_mean"]

    # v17: 추가 scene vs current interactions
    if "order_inflow_15m" in df.columns and "scene_order_inflow_15m_mean" in df.columns:
        df["order_vs_scene_mean"]  = df["order_inflow_15m"] - df["scene_order_inflow_15m_mean"]
        df["order_vs_scene_ratio"] = df["order_inflow_15m"] / (df["scene_order_inflow_15m_mean"] + EPS)

    if "robot_utilization" in df.columns and "scene_robot_utilization_mean" in df.columns:
        df["utilization_vs_scene_mean"] = df["robot_utilization"] - df["scene_robot_utilization_mean"]

    if "low_battery_ratio" in df.columns and "scene_low_battery_ratio_mean" in df.columns:
        df["battery_vs_scene_mean"] = df["low_battery_ratio"] - df["scene_low_battery_ratio_mean"]

    if "avg_trip_distance" in df.columns and "scene_avg_trip_distance_mean" in df.columns:
        df["trip_vs_scene_mean"]  = df["avg_trip_distance"] - df["scene_avg_trip_distance_mean"]
        df["trip_vs_scene_ratio"] = df["avg_trip_distance"] / (df["scene_avg_trip_distance_mean"] + EPS)

    # v18: top scene features × availability_ratio_log1p
    if "availability_ratio_log1p" in df.columns:
        for scene_col, short in [
            ("scene_congestion_score_mean",   "congestion"),
            ("scene_max_zone_density_mean",   "density"),
            ("scene_near_collision_15m_mean", "collision"),
            ("scene_avg_trip_distance_mean",  "trip"),
        ]:
            if scene_col in df.columns:
                _s(f"scene_{short}_x_avail", _p(scene_col, "availability_ratio_log1p"))

    return df


train = make_features(train)
test  = make_features(test)
print(f"after FE: train {train.shape}, test {test.shape}")


# =========================================================
# Target & feature split
# =========================================================
y_raw  = train[TARGET].copy()
y_sqrt = pd.Series(t_sqrt(y_raw), index=y_raw.index)

feat_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
X = train[feat_cols].copy()
X_test = test[feat_cols].copy()
print(f"features: {len(feat_cols)}")

groups = train["layout_id"].astype(str) + "_" + train["scenario_id"].astype(str)
gkf = GroupKFold(n_splits=N_SPLITS)


# =========================================================
# OOF Target Encoding (v18: Bayesian smoothing 적용)
# smoothing=10: 샘플 수 적은 카테고리의 TE 분산 억제
# =========================================================
TE_SMOOTHING = 10

def _smooth_enc(y_tr, key_tr, gmean, smoothing=TE_SMOOTHING):
    stats = y_tr.groupby(key_tr).agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * gmean) / (stats["count"] + smoothing)
    return smooth

def add_oof_te(X, X_test, y, splitter, groups):
    X, X_test = X.copy(), X_test.copy()
    gmean = float(y.mean())

    te_keys = ["layout_id", "scenario_id", "layout_type", "layout_cluster"]
    for key in [c for c in te_keys if c in X.columns]:
        col = f"te__{key}"
        X[col] = np.nan
        enc_all = _smooth_enc(y, X[key], gmean)
        X_test[col] = X_test[key].map(enc_all).fillna(gmean)
        for tr_i, va_i in splitter.split(X, y, groups=groups):
            enc = _smooth_enc(y.iloc[tr_i], X[key].iloc[tr_i], gmean)
            X.iloc[va_i, X.columns.get_loc(col)] = (
                X[key].iloc[va_i].map(enc).fillna(gmean).values
            )

    # layout × scenario combo TE
    if "layout_id" in X.columns and "scenario_id" in X.columns:
        ck   = X["layout_id"].astype(str) + "__" + X["scenario_id"].astype(str)
        ck_t = X_test["layout_id"].astype(str) + "__" + X_test["scenario_id"].astype(str)
        col = "te__layout_scenario"
        X[col] = np.nan
        enc_all = _smooth_enc(y, ck, gmean)
        X_test[col] = ck_t.map(enc_all).fillna(gmean)
        for tr_i, va_i in splitter.split(X, y, groups=groups):
            enc = _smooth_enc(y.iloc[tr_i], ck.iloc[tr_i], gmean)
            X.iloc[va_i, X.columns.get_loc(col)] = ck.iloc[va_i].map(enc).fillna(gmean).values

    # layout_cluster × layout_type combo TE
    if "layout_cluster" in X.columns and "layout_type" in X.columns:
        ck   = X["layout_cluster"].astype(str) + "__" + X["layout_type"].astype(str)
        ck_t = X_test["layout_cluster"].astype(str) + "__" + X_test["layout_type"].astype(str)
        col = "te__cluster_type"
        X[col] = np.nan
        enc_all = _smooth_enc(y, ck, gmean)
        X_test[col] = ck_t.map(enc_all).fillna(gmean)
        for tr_i, va_i in splitter.split(X, y, groups=groups):
            enc = _smooth_enc(y.iloc[tr_i], ck.iloc[tr_i], gmean)
            X.iloc[va_i, X.columns.get_loc(col)] = ck.iloc[va_i].map(enc).fillna(gmean).values

    return X, X_test


X, X_test = add_oof_te(X, X_test, y_raw, gkf, groups)
print(f"features after TE: {X.shape[1]}")


# =========================================================
# Categorical encoding
# =========================================================
cat_cols = [c for c in ["layout_id", "scenario_id", "layout_type", "layout_cluster"]
            if c in X.columns]
for c in X.select_dtypes("object").columns:
    if c not in cat_cols:
        cat_cols.append(c)

full = pd.concat([X, X_test], ignore_index=True)
for col in cat_cols:
    full[col] = full[col].fillna("__MISSING__").astype(str)
    mapping = {v: i for i, v in enumerate(full[col].unique())}
    full[col] = full[col].map(mapping).astype("int32")

n_tr = len(X)
X = full.iloc[:n_tr].copy()
X_test = full.iloc[n_tr:].copy()
print("encoded cat cols:", cat_cols)


# =========================================================
# Model base params
# =========================================================
XGB_BASE = dict(
    objective="reg:absoluteerror", eval_metric="mae",
    tree_method="hist", device="cuda",
    random_state=RANDOM_STATE, n_jobs=0,
    early_stopping_rounds=100, verbosity=0,
)
LGB_BASE = dict(
    objective="mae", metric="mae",
    device="cpu",
    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
)


# =========================================================
# Optuna – XGBoost
# =========================================================
def xgb_trial(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 3000, 10000, step=250),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        max_depth        = trial.suggest_int("max_depth", 5, 11),
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        subsample        = trial.suggest_float("subsample", 0.65, 0.95),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.9),
        colsample_bylevel= trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        gamma            = trial.suggest_float("gamma", 0.0, 15.0),
        max_bin          = trial.suggest_int("max_bin", 128, 512, step=64),
    )
    maes = []
    for tr_i, va_i in gkf.split(X, y_sqrt, groups=groups):
        m = XGBRegressor(**XGB_BASE, **p)
        try:
            m.fit(X.iloc[tr_i], y_sqrt.iloc[tr_i],
                  eval_set=[(X.iloc[va_i], y_sqrt.iloc[va_i])], verbose=False)
        except Exception as e:
            if "CUDA_ERROR" in str(e) or "cuMem" in str(e):
                raise optuna.exceptions.TrialPruned()
            raise
        pred = inv_sqrt(np.clip(m.predict(X.iloc[va_i]), 0, None))
        maes.append(mean_absolute_error(y_raw.iloc[va_i], pred))
    return float(np.mean(maes))


def _cb(study, trial):
    print(f"  Trial {trial.number:3d} | MAE={trial.value:.6f} | best={study.best_value:.6f}", flush=True)

print("\n=== XGBoost Optuna ===")
study_xgb = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study_xgb.optimize(xgb_trial, n_trials=N_TRIALS_XGB, callbacks=[_cb])
print(f"XGB best CV: {study_xgb.best_value:.6f}")
print("XGB best params:", study_xgb.best_params)


# =========================================================
# Optuna – LightGBM
# =========================================================
def lgb_trial(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 2000, 10000, step=250),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        max_depth        = trial.suggest_int("max_depth", 5, 11),
        num_leaves       = trial.suggest_int("num_leaves", 63, 511),
        min_child_samples= trial.suggest_int("min_child_samples", 10, 150),
        subsample        = trial.suggest_float("subsample", 0.65, 0.95),
        subsample_freq   = 1,
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.35, 0.85),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        min_split_gain   = trial.suggest_float("min_split_gain", 0.0, 3.0),
    )
    maes = []
    for tr_i, va_i in gkf.split(X, y_sqrt, groups=groups):
        m = LGBMRegressor(**LGB_BASE, **p)
        m.fit(
            X.iloc[tr_i], y_sqrt.iloc[tr_i],
            eval_set=[(X.iloc[va_i], y_sqrt.iloc[va_i])],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        pred = inv_sqrt(np.clip(m.predict(X.iloc[va_i]), 0, None))
        maes.append(mean_absolute_error(y_raw.iloc[va_i], pred))
    return float(np.mean(maes))


print("\n=== LightGBM Optuna ===")
study_lgb = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study_lgb.optimize(lgb_trial, n_trials=N_TRIALS_LGB, callbacks=[_cb])
print(f"LGB best CV: {study_lgb.best_value:.6f}")
print("LGB best params:", study_lgb.best_params)


# =========================================================
# Final CV
# =========================================================
def run_final_cv(ModelCls, model_params, is_lgb=False):
    oof    = np.zeros(len(X))
    test_p = np.zeros(len(X_test))
    fi_list = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(X, y_sqrt, groups=groups), 1):
        m = ModelCls(**model_params)

        if is_lgb:
            m.fit(
                X.iloc[tr_i], y_sqrt.iloc[tr_i],
                eval_set=[(X.iloc[va_i], y_sqrt.iloc[va_i])],
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
        else:
            try:
                m.fit(
                    X.iloc[tr_i], y_sqrt.iloc[tr_i],
                    eval_set=[(X.iloc[va_i], y_sqrt.iloc[va_i])],
                    verbose=100,
                )
            except Exception as e:
                if "CUDA_ERROR" in str(e) or "cuMem" in str(e):
                    print(f"  [Fold {fold}] CUDA VMM error → CPU fallback")
                    cpu_params = {**model_params, "device": "cpu", "n_jobs": -1}
                    m = ModelCls(**cpu_params)
                    m.fit(
                        X.iloc[tr_i], y_sqrt.iloc[tr_i],
                        eval_set=[(X.iloc[va_i], y_sqrt.iloc[va_i])],
                        verbose=100,
                    )
                else:
                    raise

        oof[va_i] = inv_sqrt(np.clip(m.predict(X.iloc[va_i]), 0, None))
        test_p += inv_sqrt(np.clip(m.predict(X_test), 0, None)) / N_SPLITS

        fold_mae = mean_absolute_error(y_raw.iloc[va_i], oof[va_i])
        print(f"  Fold {fold} MAE: {fold_mae:.6f}")

        if hasattr(m, "feature_importances_"):
            fi_list.append(pd.DataFrame({
                "feature": X.columns,
                "importance": m.feature_importances_,
                "fold": fold,
            }))

    cv_mae  = mean_absolute_error(y_raw, oof)
    cv_rmse = np.sqrt(mean_squared_error(y_raw, oof))
    print(f"  CV MAE: {cv_mae:.6f}  RMSE: {cv_rmse:.6f}\n")
    return oof, test_p, fi_list


print("\n==============================")
print("XGBoost Final CV")
print("==============================")
xgb_full_params = {**XGB_BASE, **study_xgb.best_params}
xgb_oof, xgb_test, xgb_fi = run_final_cv(XGBRegressor, xgb_full_params, is_lgb=False)

print("==============================")
print("LightGBM Final CV")
print("==============================")
lgb_full_params = {**LGB_BASE, **study_lgb.best_params}
lgb_oof, lgb_test, lgb_fi = run_final_cv(LGBMRegressor, lgb_full_params, is_lgb=True)


# =========================================================
# Optimal blend (grid search over XGB weight)
# =========================================================
cv_xgb = mean_absolute_error(y_raw, xgb_oof)
cv_lgb = mean_absolute_error(y_raw, lgb_oof)
print(f"XGB CV MAE : {cv_xgb:.6f}")
print(f"LGB CV MAE : {cv_lgb:.6f}")

best_w, best_cv = 0.5, float("inf")
for w in np.arange(0.0, 1.01, 0.02):
    mae = mean_absolute_error(y_raw, w * xgb_oof + (1 - w) * lgb_oof)
    if mae < best_cv:
        best_cv, best_w = mae, w

print(f"Optimal XGB weight: {best_w:.2f}  →  Blend CV MAE: {best_cv:.6f}")

final_pred = np.clip(best_w * xgb_test + (1 - best_w) * lgb_test, 0, None)


# =========================================================
# Feature importance
# =========================================================
for tag, fi_list in [("xgb", xgb_fi), ("lgb", lgb_fi)]:
    if fi_list:
        fi_df = pd.concat(fi_list, ignore_index=True)
        fi_mean = (
            fi_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fi_mean.to_csv(DATA_DIR / f"feature_importance_{tag}_v5.csv", index=False)
        print(f"\nTop 20 {tag.upper()} features:")
        print(fi_mean.head(20).to_string(index=False))


# =========================================================
# OOF diagnostics
# =========================================================
oof_blend = best_w * xgb_oof + (1 - best_w) * lgb_oof
oof_df = train[[ID_COL, "layout_id", "scenario_id"]].copy()
oof_df["y_true"]       = y_raw.values
oof_df["y_pred_xgb"]   = xgb_oof
oof_df["y_pred_lgb"]   = lgb_oof
oof_df["y_pred_blend"] = oof_blend
oof_df["abs_err"]      = np.abs(oof_df["y_true"] - oof_df["y_pred_blend"])
oof_df.to_csv(DATA_DIR / "oof_ensemble_v5.csv", index=False)

layout_mae = (
    oof_df.groupby("layout_id")["abs_err"]
    .mean()
    .sort_values(ascending=False)
)
print("\nTop 10 worst layouts (blend):")
print(layout_mae.head(10))


# =========================================================
# Submission
# =========================================================
submission = sample_sub.copy()
submission[TARGET] = final_pred
submission.to_csv(DATA_DIR / "submission_ensemble_v5.csv", index=False)

print("\nSaved: submission_ensemble_v5.csv")
print("Saved: oof_ensemble_v5.csv")
print("Saved: feature_importance_xgb_v3.csv")
print("Saved: feature_importance_lgb_v3.csv")


# =========================================================
# Sanity check
# =========================================================
print("\n=== Sanity Check ===")
for name, arr in [("y_raw", y_raw), ("xgb_oof", xgb_oof),
                  ("lgb_oof", lgb_oof), ("final_pred", final_pred)]:
    s = pd.Series(arr)
    print(f"{name:15s}  min={s.min():.3f}  mean={s.mean():.3f}  "
          f"max={s.max():.3f}  nan={s.isna().sum()}")
