# -*- coding: utf-8 -*-
from pathlib import Path
import gc
import os
import platform
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

DATA_DIR = Path(".")
OUTPUT_DIR = Path(".")
TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]

N_SPLITS = 10
BASE_SEED = 42

SEED_BANK_FIXED = [42, 2024, 7777, 100, 3407, 11, 777, 999, 1234, 5555]
_seed_rng = np.random.default_rng(20280502)
_extra_seeds = []
while len(SEED_BANK_FIXED) + len(_extra_seeds) < 100:
    s = int(_seed_rng.integers(1, 2_000_000_000))
    if s not in SEED_BANK_FIXED and s not in _extra_seeds:
        _extra_seeds.append(s)

SEED_BANK_100 = SEED_BANK_FIXED + _extra_seeds
MODEL_SEED_COUNT = 10
MODEL_SEEDS = SEED_BANK_100[:MODEL_SEED_COUNT]
SEEDS_10 = MODEL_SEEDS
SEED_COUNTS = [MODEL_SEED_COUNT]

LGB_N_ESTIMATORS = 5000
LGB_EARLY_STOPPING = 100
SCIPY_TRIALS = 80
HAS_CUDA = torch.cuda.is_available()

ENVIRONMENT = {
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "lightgbm": lgb.__version__,
    "torch": torch.__version__,
    "cuda_available": str(HAS_CUDA),
}

NOISE_COLS = [
    "warehouse_temp_avg", "humidity_pct", "external_temp_c", "wind_speed_kmh",
    "precipitation_mm", "lighting_level_lux", "ambient_noise_db", "floor_vibration_idx",
    "return_order_ratio", "air_quality_idx", "co2_level_ppm", "hvac_power_kw",
    "wms_response_time_ms", "scanner_error_rate", "wifi_signal_db", "network_latency_ms",
    "worker_avg_tenure_months", "safety_score_monthly", "label_print_queue",
    "barcode_read_success_rate", "ups_battery_pct", "lighting_zone_variance",
    "robot_calibration_score", "racking_height_avg_m", "shift_handover_delay_min",
]

SHAP_REMOVE = [
    "floor_area_sqm", "fire_sprinkler_count", "ceiling_height_m",
    "has_charge_queue", "has_blocked_path", "battery_crisis", "recovery_missing",
    "is_late_shift", "task_reassign_15m", "replenishment_overlap", "has_fault",
    "pallet_wrap_time_min", "avg_charge_wait", "storage_density_pct", "avg_recovery_time",
    "day_of_week", "fault_count_15m", "building_age_years",
]

QUEUE_COLS_V2 = [
    "pack_rho", "pack_explosion_log", "pack_wait_mm1", "near_pack_saturation",
    "pack_sat_intensity", "robot_total_eff", "robot_rho", "robot_explosion_log",
    "low_battery_count_est", "charger_rho", "charger_explosion_log", "bottleneck_rho",
    "bottleneck_explosion_log", "n_critical_resources", "order_wip_proxy",
    "order_cv_squared", "kingman_wait_pack", "kingman_wait_log", "burst_x_rho",
    "burst_x_explosion",
]

SCEN_REL_BASE_COLS = [
    "order_inflow_15m", "congestion_score", "low_battery_ratio", "pack_utilization",
    "robot_utilization", "robot_active", "robot_idle", "robot_charging", "charging_ratio",
    "order_per_pack", "order_per_robot", "order_per_active_robot", "multi_resource_stress",
    "pack_queue_pressure", "robot_slack", "effective_robot", "order_surge_ratio",
]

SCEN_REL_QUEUE_COLS = [
    "pack_rho", "robot_rho", "charger_rho", "bottleneck_rho", "pack_wait_mm1",
    "kingman_wait_log", "burst_x_explosion", "order_wip_proxy",
]

DEEP_EPOCHS = 150
DEEP_PATIENCE = 15
DEEP_BATCH = 64
DEEP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE2_LIKE_COLS = [
    "layout_target_mean", "layout_target_std", "layout_target_median",
    "pseudo_delay", "cum_pseudo", "pseudo_lag1",
    "cum_pred", "max_pred_so_far", "expanding_mean_pred",
    "pred_lag1_log", "pred_rolling3", "pred_diff1",
    "pred_p_high", "pred_mean_delay", "pred_max_delay", "pred_pct90",
]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def seed_everything_torch(seed=42):
    seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def make_lgb_params(seed=42):
    return dict(
        device="gpu" if HAS_CUDA else "cpu",
        objective="mae",
        n_estimators=LGB_N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        min_child_samples=20,
        random_state=seed,
        verbose=-1,
    )


def make_lgb_small_params(seed=42):
    return dict(
        device="gpu" if HAS_CUDA else "cpu",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=20,
        random_state=seed,
        verbose=-1,
    )


def cc(s):
    a = np.asarray(s)
    p = (a > 0).astype(np.int32)
    r = np.zeros(len(a), dtype=np.int32)
    for i in range(len(a)):
        r[i] = r[i - 1] + 1 if p[i] and i > 0 else p[i]
    return r


def stage1_fe(df, lay, raw=None, bt=None):
    df = df.copy().merge(lay, on="layout_id", how="left")
    df.drop(columns=[c for c in NOISE_COLS if c in df.columns], inplace=True)
    df.sort_values(["scenario_id", "ID"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    g = df.groupby("scenario_id")
    df["timeslot"] = g.cumcount()

    df["is_congested"] = (df["congestion_score"] > 0).astype(np.int8)
    df["has_charging"] = (df["robot_charging"] > 0).astype(np.int8)

    df["robot_total_calc"] = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    df["order_per_pack"] = df["order_inflow_15m"] / df["pack_station_count"]
    df["charging_ratio"] = df["robot_charging"] / (df["robot_total_calc"] + 1)
    df["order_per_robot"] = df["order_inflow_15m"] / (df["robot_total_calc"] + 1)
    df["util_per_pack"] = df["robot_utilization"] * df["robot_total_calc"] / df["pack_station_count"]
    df["order_per_active_robot"] = df["order_inflow_15m"] / (df["robot_active"] + 1)

    df["pack_bottleneck_manual"] = df["order_per_pack"] * (1 - df["manual_override_ratio"].fillna(0))
    df["battery_congestion_spiral"] = df["low_battery_ratio"] * df["congestion_score"]
    df["wave_charge_conflict"] = df["order_wave_count"] * df["charging_ratio"]

    df["is_warmup"] = (df["timeslot"] <= 3).astype(np.int8)

    for c in ["order_inflow_15m", "congestion_score", "low_battery_ratio", "robot_utilization"]:
        if c in df.columns:
            df[f"{c}_lag1"] = g[c].shift(1)

    for c in ["order_inflow_15m", "congestion_score", "low_battery_ratio"]:
        if c in df.columns:
            df[f"{c}_rolling3"] = g[c].transform(lambda x: x.rolling(3, 1).mean())

    df["order_inflow_diff1"] = g["order_inflow_15m"].diff(1)
    df["congestion_diff1"] = g["congestion_score"].diff(1)

    df["congestion_roll3_x_charging"] = df["congestion_score_rolling3"] * df["charging_ratio"]
    df["congestion_roll3_x_order_pack"] = df["congestion_score_rolling3"] * df["order_per_pack"]
    df["idle_x_charging"] = df["robot_idle"] * df["charging_ratio"]
    df["pack_util_x_order_pack"] = df["pack_utilization"] * df["order_per_pack"]

    df["congestion_trend"] = df["congestion_score"] / (df["congestion_score_rolling3"] + 0.01)
    df["battery_trend"] = df["low_battery_ratio"] / (df["low_battery_ratio_rolling3"] + 0.01)
    df["order_acceleration"] = g["order_inflow_diff1"].diff(1)

    if raw is not None:
        oc = [c for c in raw.columns if c not in ID_COLS + [TARGET] + NOISE_COLS]
        rs = raw.copy().merge(lay[["layout_id"]], on="layout_id", how="left")
        rs = rs.sort_values(["scenario_id", "ID"]).reset_index(drop=True)
        df["missing_count"] = rs[oc].isnull().sum(axis=1).values
    else:
        df["missing_count"] = 0

    df["robot_slack"] = df["robot_idle"] / (df["robot_total_calc"] + 1)
    df["slack_expanding_min"] = g["robot_slack"].transform(lambda x: x.expanding().min())
    df["max_congestion_so_far"] = g["congestion_score"].transform(lambda x: x.expanding().max())

    df["pack_queue_pressure"] = df["order_inflow_15m"] / (
        df["pack_station_count"] * (1.01 - df["pack_utilization"].clip(0, 0.99))
    )

    df["multi_resource_stress"] = (
        df["congestion_score"] / 100
        + df["low_battery_ratio"]
        + df["robot_utilization"]
        + df["pack_utilization"]
    )
    if "charge_queue_length" in df.columns:
        df["multi_resource_stress"] += df["charge_queue_length"] / 30

    df["consec_congestion"] = g["congestion_score"].transform(lambda x: cc(x.values))
    df["order_inflow_std3"] = g["order_inflow_15m"].transform(lambda x: x.rolling(3, 1).std())
    df["cum_congestion"] = g["congestion_score"].cumsum()

    if "charge_queue_length" in df.columns and "charger_count" in df.columns:
        df["queue_per_charger"] = df["charge_queue_length"] / (df["charger_count"] + 1)

    df["cong_frac_so_far"] = g["congestion_score"].transform(lambda x: (x > 0).expanding().mean())
    df["cum_battery_stress"] = g["low_battery_ratio"].cumsum()
    df["max_order_so_far"] = g["order_inflow_15m"].transform(lambda x: x.expanding().max())
    df["max_pack_util_so_far"] = g["pack_utilization"].transform(lambda x: x.expanding().max())
    df["min_battery_so_far"] = g["low_battery_ratio"].transform(lambda x: x.expanding().min())

    if bt is None:
        bt = {
            "congestion_score": df["congestion_score"].quantile(0.75),
            "low_battery_ratio": df["low_battery_ratio"].quantile(0.75),
            "pack_utilization": df["pack_utilization"].quantile(0.75),
            "charging_ratio": df["charging_ratio"].quantile(0.75),
        }

    df["n_bottlenecks"] = sum((df[c] > t).astype(int) for c, t in bt.items())

    df["expanding_mean_cong"] = g["congestion_score"].transform(lambda x: x.expanding().mean())
    df["expanding_mean_battery"] = g["low_battery_ratio"].transform(lambda x: x.expanding().mean())
    df["expanding_mean_order"] = g["order_inflow_15m"].transform(lambda x: x.expanding().mean())
    df["max_util_so_far"] = g["robot_utilization"].transform(lambda x: x.expanding().max())
    df["max_charging_so_far"] = g["charging_ratio"].transform(lambda x: x.expanding().max())

    df["cum_pack_util"] = g["pack_utilization"].cumsum()
    df["cum_stress"] = g["multi_resource_stress"].cumsum()
    df["effective_robot"] = df["robot_active"] * (1 - df["low_battery_ratio"])
    df["order_surge_ratio"] = df["order_inflow_15m"] / (df["order_inflow_15m_rolling3"] + 1)

    mt = g["timeslot"].transform("max").clip(1)
    df["scenario_progress"] = df["timeslot"] / mt

    df["late_cum_congestion"] = df["scenario_progress"] * df["cum_congestion"]
    df["pack_headroom"] = df["pack_station_count"] * (1 - df["pack_utilization"])
    df["cum_cong_x_cum_batt"] = df["cum_congestion"] * df["cum_battery_stress"]
    df["max_cong_x_cong_frac"] = df["max_congestion_so_far"] * df["cong_frac_so_far"]

    df["congestion_recovery"] = df["congestion_score"] / (df["max_congestion_so_far"] + 0.01)
    df["battery_recovery"] = df["low_battery_ratio"] / (
        g["low_battery_ratio"].transform(lambda x: x.expanding().max()) + 0.01
    )
    df["order_load_ratio"] = df["order_inflow_15m"] / (df["max_order_so_far"] + 1)

    df["expanding_std_cong"] = g["congestion_score"].transform(lambda x: x.expanding().std().fillna(0))
    df["expanding_std_order"] = g["order_inflow_15m"].transform(lambda x: x.expanding().std().fillna(0))
    df["expanding_std_battery"] = g["low_battery_ratio"].transform(lambda x: x.expanding().std().fillna(0))

    df.drop(columns=[c for c in SHAP_REMOVE if c in df.columns], inplace=True)
    return df, bt


def add_queueing_features_v2(df):
    df = df.copy()
    eps = 1e-6

    df["pack_rho"] = df["pack_utilization"].clip(0, 0.999)
    df["pack_explosion_log"] = np.log1p(1.0 / (1.0 - df["pack_rho"] + eps))
    df["pack_wait_mm1"] = df["pack_rho"] ** 2 / (1.0 - df["pack_rho"] + eps)
    df["near_pack_saturation"] = (df["pack_rho"] > 0.85).astype(np.int8)
    df["pack_sat_intensity"] = (df["pack_rho"] - 0.85).clip(0, None) ** 2

    df["robot_total_eff"] = df["robot_active"] * (1.0 - df["low_battery_ratio"])
    df["robot_rho"] = (df["order_inflow_15m"] / (df["robot_total_eff"] * 5.0 + eps)).clip(0, 0.999)
    df["robot_explosion_log"] = np.log1p(1.0 / (1.0 - df["robot_rho"] + eps))

    df["low_battery_count_est"] = df["low_battery_ratio"] * df["robot_total_calc"]
    df["charger_rho"] = (df["low_battery_count_est"] / (df["charger_count"] * 2.0 + eps)).clip(0, 0.999)
    df["charger_explosion_log"] = np.log1p(1.0 / (1.0 - df["charger_rho"] + eps))

    rho_cols = ["pack_rho", "robot_rho", "charger_rho"]
    df["bottleneck_rho"] = df[rho_cols].max(axis=1)
    df["bottleneck_explosion_log"] = np.log1p(1.0 / (1.0 - df["bottleneck_rho"].clip(0, 0.999) + eps))
    df["n_critical_resources"] = (df[rho_cols] > 0.85).sum(axis=1).astype(np.int8)
    df["order_wip_proxy"] = df["order_inflow_15m"] * df["pack_wait_mm1"]

    order_mean3 = df["order_inflow_15m_rolling3"].clip(eps, None)
    order_cv = (df["order_inflow_std3"].fillna(0) / order_mean3).clip(0, 5)
    df["order_cv_squared"] = order_cv ** 2

    df["kingman_wait_pack"] = (df["pack_rho"] / (1.0 - df["pack_rho"] + eps)) * (1.0 + df["order_cv_squared"]) / 2.0
    df["kingman_wait_log"] = np.log1p(df["kingman_wait_pack"])

    df["burst_x_rho"] = df["order_cv_squared"] * df["pack_rho"]
    df["burst_x_explosion"] = df["order_cv_squared"] * df["pack_explosion_log"]
    return df


def add_scenario_relative_features(df, include_queue_cols=True):
    df = df.copy()
    g = df.groupby("scenario_id")

    rel_cols = list(SCEN_REL_BASE_COLS)
    if include_queue_cols:
        rel_cols += SCEN_REL_QUEUE_COLS

    rel_cols = [c for c in rel_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    for c in rel_cols:
        mean = g[c].transform("mean")
        med = g[c].transform("median")
        std = g[c].transform("std").replace(0, np.nan)
        df[f"{c}_vs_scen_mean"] = df[c] - mean
        df[f"{c}_vs_scen_median"] = df[c] - med
        df[f"{c}_scen_z"] = ((df[c] - mean) / (std + 1e-6)).replace([np.inf, -np.inf], 0).fillna(0)
        df[f"{c}_scen_rank"] = g[c].rank(pct=True).fillna(0.5)

    return df


def mae_bucket_report(y, p):
    out = {"MAE_all": mean_absolute_error(y, p)}
    for lo, hi, name in [(0, 10, "0_10"), (10, 50, "10_50"), (50, 999999, "50_plus")]:
        m = (y >= lo) & (y < hi)
        out[f"MAE_{name}"] = mean_absolute_error(y[m], p[m]) if m.sum() else np.nan
    return out


def add_result_row(rows, exp_name, model_type, seed_mode, fe_name, oof_pred, y_true, extra=None):
    b = mae_bucket_report(y_true, oof_pred)
    row = {
        "exp_name": exp_name,
        "model_type": model_type,
        "seed_mode": seed_mode,
        "fe_name": fe_name,
        "oof_mae": b["MAE_all"],
        "MAE_0_10": b["MAE_0_10"],
        "MAE_10_50": b["MAE_10_50"],
        "MAE_50_plus": b["MAE_50_plus"],
    }
    if extra:
        row.update(extra)
    rows.append(row)


def scipy_weight_ensemble(oof_dict, y_true, n_trials=80, seed=42):
    names = list(oof_dict.keys())
    mat = np.column_stack([np.clip(oof_dict[n], 0, None) for n in names])

    def obj(w):
        w = np.abs(w)
        w = w / max(w.sum(), 1e-12)
        return mean_absolute_error(y_true, mat @ w)

    rng = np.random.default_rng(seed)
    best_w = None
    best_e = float("inf")

    for trial in range(n_trials):
        if trial == 0:
            w0 = np.ones(len(names)) / len(names)
        else:
            w0 = rng.dirichlet(np.ones(len(names)))

        res = minimize(
            obj,
            w0,
            method="Nelder-Mead",
            options={"maxiter": 30000, "xatol": 1e-8, "fatol": 1e-8},
        )

        if res.fun < best_e:
            best_e = float(res.fun)
            best_w = np.abs(res.x)
            best_w = best_w / best_w.sum()

    ens = mat @ best_w
    return names, best_w, best_e, ens


def make_feature_version(train_stage1, test_stage1, with_queue=False, with_rel=False):
    train = train_stage1.copy()
    test = test_stage1.copy()

    if with_queue:
        train = add_queueing_features_v2(train)
        test = add_queueing_features_v2(test)

    if with_rel:
        train = add_scenario_relative_features(train, include_queue_cols=with_queue)
        test = add_scenario_relative_features(test, include_queue_cols=with_queue)

    train["layout_type"] = train["layout_type"].astype("category")
    test["layout_type"] = test["layout_type"].astype("category")
    return train, test


def scenario_agg_table(df, y_raw=None, folds=None):
    base_cols = [
        "order_per_pack", "congestion_score", "low_battery_ratio", "pack_utilization",
        "robot_utilization", "charging_ratio", "multi_resource_stress", "order_inflow_15m",
        "robot_active", "robot_idle", "pack_station_count", "timeslot",
        "pack_rho", "robot_rho", "charger_rho", "bottleneck_rho", "kingman_wait_log",
        "burst_x_explosion", "order_wip_proxy", "congestion_score_scen_z",
        "order_inflow_15m_scen_rank", "pack_utilization_scen_z", "bottleneck_rho_scen_rank",
    ]
    agg_cols = [c for c in base_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    agg_dict = {}
    for c in agg_cols:
        if c == "pack_station_count":
            agg_dict[c] = ["first"]
        elif c == "timeslot":
            agg_dict[c] = ["max"]
        else:
            agg_dict[c] = ["mean", "std", "max", "min"]

    sf = df.groupby("scenario_id").agg(agg_dict).reset_index()
    sf.columns = ["scenario_id"] + [f"sf_{a}_{b}" for a, b in sf.columns[1:]]

    early_cols = [c for c in agg_cols if c not in ["timeslot", "pack_station_count"]]
    early = df[df["timeslot"] <= 2].groupby("scenario_id")[early_cols].agg(["mean", "max"]).reset_index()
    early.columns = ["scenario_id"] + [f"init_{a}_{b}" for a, b in early.columns[1:]]
    sf = sf.merge(early, on="scenario_id", how="left")

    if y_raw is not None and folds is not None:
        gy = pd.DataFrame({"scenario_id": df["scenario_id"].values, "y": y_raw})
        a = gy.groupby("scenario_id")["y"]
        st = a.agg(["max", "mean"]).reset_index()
        st.columns = ["scenario_id", "mx", "mn"]
        st["h"] = (st["mx"] > 50).astype(int)
        st["p90"] = a.quantile(0.9).values
        sf = sf.merge(st, on="scenario_id", how="left")

        s2f = {}
        for fi, (_, vi) in enumerate(folds):
            for sid in df.iloc[vi]["scenario_id"].unique():
                s2f[sid] = fi
        sf["_f"] = sf["scenario_id"].map(s2f)

    return sf


def build_lgb_stage2_dataset(train_stage1, test_stage1, y_raw, folds, with_queue=False, with_rel=False, seed=42, fe_name="BASE"):
    train, test = make_feature_version(train_stage1, test_stage1, with_queue=with_queue, with_rel=with_rel)
    train["layout_type"] = train["layout_type"].astype("category")
    test["layout_type"] = test["layout_type"].astype("category")

    y_log = np.log1p(y_raw)
    s1fc = [c for c in train.columns if c not in ID_COLS + [TARGET]]

    sd = scenario_agg_table(train, y_raw=y_raw, folds=folds)
    test_sf = scenario_agg_table(test, y_raw=None, folds=None)

    sfc = [c for c in sd.columns if c.startswith("sf_") or c.startswith("init_")]
    sfc = [c for c in sfc if c in test_sf.columns]

    for pc in ["pred_p_high", "pred_mean_delay", "pred_max_delay", "pred_pct90"]:
        sd[pc] = np.nan

    small_params = make_lgb_small_params(seed)

    for fi in range(N_SPLITS):
        tr = sd["_f"] != fi
        vl = sd["_f"] == fi
        Xtr = sd.loc[tr, sfc].values
        Xvl = sd.loc[vl, sfc].values

        m = lgb.LGBMClassifier(objective="binary", **small_params)
        m.fit(Xtr, sd.loc[tr, "h"].values)
        sd.loc[vl, "pred_p_high"] = m.predict_proba(Xvl)[:, 1]
        del m

        m = lgb.LGBMRegressor(objective="mae", **small_params)
        m.fit(Xtr, sd.loc[tr, "mn"].values)
        sd.loc[vl, "pred_mean_delay"] = np.clip(m.predict(Xvl), 0, None)
        del m

        m = lgb.LGBMRegressor(objective="mae", **small_params)
        m.fit(Xtr, np.log1p(sd.loc[tr, "mx"].values))
        sd.loc[vl, "pred_max_delay"] = np.expm1(m.predict(Xvl))
        del m

        m = lgb.LGBMRegressor(objective="mae", **small_params)
        m.fit(Xtr, np.log1p(sd.loc[tr, "p90"].values))
        sd.loc[vl, "pred_pct90"] = np.expm1(m.predict(Xvl))
        del m
        gc.collect()

    try:
        v15_auc = roc_auc_score(sd["h"], sd["pred_p_high"])
    except Exception:
        v15_auc = np.nan

    Xtr_all = sd[sfc].values
    Xte_sf = test_sf[sfc].values

    m = lgb.LGBMClassifier(objective="binary", **small_params)
    m.fit(Xtr_all, sd["h"].values)
    test_sf["pred_p_high"] = m.predict_proba(Xte_sf)[:, 1]
    del m

    m = lgb.LGBMRegressor(objective="mae", **small_params)
    m.fit(Xtr_all, sd["mn"].values)
    test_sf["pred_mean_delay"] = np.clip(m.predict(Xte_sf), 0, None)
    del m

    m = lgb.LGBMRegressor(objective="mae", **small_params)
    m.fit(Xtr_all, np.log1p(sd["mx"].values))
    test_sf["pred_max_delay"] = np.expm1(m.predict(Xte_sf))
    del m

    m = lgb.LGBMRegressor(objective="mae", **small_params)
    m.fit(Xtr_all, np.log1p(sd["p90"].values))
    test_sf["pred_pct90"] = np.expm1(m.predict(Xte_sf))
    del m
    gc.collect()

    sm_tr = sd.set_index("scenario_id")
    sm_te = test_sf.set_index("scenario_id")

    for c in ["pred_p_high", "pred_mean_delay", "pred_max_delay", "pred_pct90"]:
        train[c] = train["scenario_id"].map(sm_tr[c])
        test[c] = test["scenario_id"].map(sm_te[c])

    s1nv = [c for c in s1fc if c not in ["pred_p_high", "pred_mean_delay", "pred_max_delay", "pred_pct90"]]
    oof_lp = np.zeros(len(train))
    tst_lp = np.zeros(len(test))
    params = make_lgb_params(seed)

    for fold, (ti, vi) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            train.loc[ti, s1nv],
            y_log[ti],
            eval_set=[(train.loc[vi, s1nv], y_log[vi])],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING), lgb.log_evaluation(0)],
        )
        oof_lp[vi] = np.clip(np.expm1(m.predict(train.loc[vi, s1nv])), 0, None)
        tst_lp += np.clip(np.expm1(m.predict(test[s1nv])), 0, None) / N_SPLITS
        del m
        gc.collect()

    pass1_mae = mean_absolute_error(y_raw, oof_lp)

    for df_, p1 in [(train, oof_lp), (test, tst_lp)]:
        df_["_p1"] = p1
        g = df_.groupby("scenario_id")["_p1"]
        df_["pred_lag1_log"] = np.log1p(g.shift(1).fillna(0))
        df_["pred_rolling3"] = g.transform(lambda x: x.shift(1).rolling(3, 1).mean()).fillna(0)
        df_["pred_diff1"] = g.diff(1).fillna(0)
        df_["cum_pred"] = g.cumsum()
        df_["max_pred_so_far"] = g.transform(lambda x: x.expanding().max())
        df_["expanding_mean_pred"] = g.transform(lambda x: x.expanding().mean())
        df_.drop(columns=["_p1"], inplace=True)

    train["layout_target_mean"] = np.nan
    train["layout_target_std"] = np.nan
    train["layout_target_median"] = np.nan

    for fold, (ti, vi) in enumerate(folds):
        fs = train.iloc[ti].groupby("layout_id")[TARGET].agg(["mean", "std", "median"])
        vl = train.iloc[vi]["layout_id"]
        idx = train.index[vi]
        train.loc[idx, "layout_target_mean"] = vl.map(fs["mean"]).values
        train.loc[idx, "layout_target_std"] = vl.map(fs["std"]).values
        train.loc[idx, "layout_target_median"] = vl.map(fs["median"]).values

    train["layout_target_mean"].fillna(y_raw.mean(), inplace=True)
    train["layout_target_std"].fillna(y_raw.std(), inplace=True)
    train["layout_target_median"].fillna(np.median(y_raw), inplace=True)

    te_s = train.groupby("layout_id")[TARGET].agg(["mean", "std", "median"])
    test["layout_target_mean"] = test["layout_id"].map(te_s["mean"]).fillna(y_raw.mean())
    test["layout_target_std"] = test["layout_id"].map(te_s["std"]).fillna(y_raw.std())
    test["layout_target_median"] = test["layout_id"].map(te_s["median"]).fillna(np.median(y_raw))

    for df_ in [train, test]:
        df_["pseudo_delay"] = (
            df_["order_per_pack"]
            * (1 + df_["congestion_score"] / 30)
            * (1 + df_["low_battery_ratio"] * 2)
            * (1 + df_["pack_utilization"])
        )
        df_["cum_pseudo"] = df_.groupby("scenario_id")["pseudo_delay"].cumsum()
        df_["pseudo_lag1"] = df_.groupby("scenario_id")["pseudo_delay"].shift(1).fillna(0)

    train["layout_type"] = train["layout_type"].astype("category")
    test["layout_type"] = test["layout_type"].astype("category")
    fc = [c for c in train.columns if c not in ID_COLS + [TARGET]]

    info = {
        "fe_name": fe_name,
        "with_queue": with_queue,
        "with_rel": with_rel,
        "v15_auc": v15_auc,
        "pass1_mae": pass1_mae,
        "stage2_feature_count": len(fc),
    }
    return train, test, fc, info


def train_lgb_sqrt_only(train, test, fc, y_raw, folds, seed=42, label="LGB_SQRT"):
    y_sqrt = np.sqrt(y_raw)
    inv = lambda p: np.clip(p, 0, None) ** 2
    params = make_lgb_params(seed)
    oof = np.zeros(len(train))
    tst = np.zeros(len(test))

    for fold, (ti, vi) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            train.loc[ti, fc],
            y_sqrt[ti],
            eval_set=[(train.loc[vi, fc], y_sqrt[vi])],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING), lgb.log_evaluation(0)],
        )
        oof[vi] = inv(m.predict(train.loc[vi, fc]))
        tst += inv(m.predict(test[fc])) / N_SPLITS
        del m
        gc.collect()

    b = mae_bucket_report(y_raw, oof)
    info = {
        "target_transform": "sqrt",
        "oof_mae": b["MAE_all"],
        "single_transform_rows": [{
            "label": label,
            "seed": seed,
            "transform": "sqrt",
            "oof_mae": b["MAE_all"],
            "MAE_0_10": b["MAE_0_10"],
            "MAE_10_50": b["MAE_10_50"],
            "MAE_50_plus": b["MAE_50_plus"],
        }],
    }
    return np.clip(oof, 0, None), np.clip(tst, 0, None), info


def add_context_features_raw(df):
    df = df.copy()
    g = df.groupby("scenario_id")
    cols = [
        "order_inflow_15m", "congestion_score", "low_battery_ratio",
        "pack_utilization", "robot_utilization", "charging_ratio",
        "pack_rho", "robot_rho", "charger_rho", "bottleneck_rho",
        "kingman_wait_log", "order_wip_proxy",
    ]
    cols = [c for c in cols if c in df.columns]
    new_cols = []

    for c in cols:
        for w in [2, 3, 5]:
            past_col = f"ctx_{c}_past{w}_mean"
            fut_col = f"ctx_{c}_future{w}_mean"
            cen_col = f"ctx_{c}_center{w}_mean"
            diff_col = f"ctx_{c}_future{w}_minus_past{w}"
            cur_col = f"ctx_{c}_cur_minus_center{w}"
            df[past_col] = g[c].transform(lambda x: x.shift(1).rolling(w, 1).mean()).fillna(df[c])
            df[fut_col] = g[c].transform(lambda x: x.shift(-1).rolling(w, 1).mean()).fillna(df[c])
            df[cen_col] = g[c].transform(lambda x: x.rolling(w, 1, center=True).mean()).fillna(df[c])
            df[diff_col] = df[fut_col] - df[past_col]
            df[cur_col] = df[c] - df[cen_col]
            new_cols += [past_col, fut_col, cen_col, diff_col, cur_col]

    return df, new_cols


def build_gru_feature_frame(train_stage1, test_stage1, context_pca_n=0):
    tr = add_queueing_features_v2(train_stage1.copy())
    te = add_queueing_features_v2(test_stage1.copy())
    tr = add_scenario_relative_features(tr, include_queue_cols=True)
    te = add_scenario_relative_features(te, include_queue_cols=True)

    pca_cols = []
    if context_pca_n > 0:
        tr, ctx_cols_tr = add_context_features_raw(tr)
        te, ctx_cols_te = add_context_features_raw(te)
        pca_cols = [c for c in ctx_cols_tr if c in ctx_cols_te]

    pca_set = set(pca_cols)
    seq_fc = [
        c for c in tr.columns
        if c not in ID_COLS + [TARGET]
        and c != "layout_type"
        and c in te.columns
        and c not in pca_set
        and pd.api.types.is_numeric_dtype(tr[c])
    ]

    bad_cols = [c for c in seq_fc if c.startswith("pred_") or c in STAGE2_LIKE_COLS]
    if bad_cols:
        raise RuntimeError(f"Deep 입력에 Stage2-like 컬럼이 섞였습니다: {bad_cols}")

    return tr, te, seq_fc, pca_cols


def prepare_deep_sequence_arrays(train_fe, test_fe, seq_fc, pca_cols):
    ts = train_fe.sort_values(["scenario_id", "ID"]).reset_index().rename(columns={"index": "orig_index"})
    te = test_fe.sort_values(["scenario_id", "ID"]).reset_index().rename(columns={"index": "orig_index"})

    if not (ts.groupby("scenario_id").size() == 25).all():
        raise ValueError("train scenario_id별 row 수가 25가 아닙니다.")
    if not (te.groupby("scenario_id").size() == 25).all():
        raise ValueError("test scenario_id별 row 수가 25가 아닙니다.")

    trs = ts["scenario_id"].unique()
    tes = te["scenario_id"].unique()
    ntr = len(trs)
    nte = len(tes)
    nf = len(seq_fc)

    X_main = np.nan_to_num(
        ts[seq_fc].replace([np.inf, -np.inf], np.nan).values.astype(np.float32).reshape(ntr, 25, nf),
        nan=0, posinf=0, neginf=0,
    )
    Xte_main = np.nan_to_num(
        te[seq_fc].replace([np.inf, -np.inf], np.nan).values.astype(np.float32).reshape(nte, 25, nf),
        nan=0, posinf=0, neginf=0,
    )

    if len(pca_cols) > 0:
        npca = len(pca_cols)
        X_pca = np.nan_to_num(
            ts[pca_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32).reshape(ntr, 25, npca),
            nan=0, posinf=0, neginf=0,
        )
        Xte_pca = np.nan_to_num(
            te[pca_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32).reshape(nte, 25, npca),
            nan=0, posinf=0, neginf=0,
        )
    else:
        X_pca = None
        Xte_pca = None

    Y_raw_seq = ts[TARGET].values.astype(np.float32).reshape(ntr, 25)
    train_idx = ts["orig_index"].values.reshape(ntr, 25)
    test_idx = te["orig_index"].values.reshape(nte, 25)
    return X_main, X_pca, Y_raw_seq, Xte_main, Xte_pca, trs, tes, train_idx, test_idx


def make_fold_deep_features(X_main, X_pca, Xte_main, Xte_pca, tr_mask, vl_mask, pca_n):
    nf_main = X_main.shape[-1]
    ntr = int(tr_mask.sum())
    nvl = int(vl_mask.sum())
    nte = Xte_main.shape[0]

    scaler_main = StandardScaler()
    Xtr_main = scaler_main.fit_transform(X_main[tr_mask].reshape(-1, nf_main)).reshape(ntr, 25, nf_main)
    Xvl_main = scaler_main.transform(X_main[vl_mask].reshape(-1, nf_main)).reshape(nvl, 25, nf_main)
    Xte_main_scaled = scaler_main.transform(Xte_main.reshape(-1, nf_main)).reshape(nte, 25, nf_main)

    if X_pca is None or pca_n <= 0:
        return Xtr_main.astype(np.float32), Xvl_main.astype(np.float32), Xte_main_scaled.astype(np.float32)

    nf_pca = X_pca.shape[-1]
    n_comp = min(int(pca_n), int(nf_pca))
    scaler_pca = StandardScaler()
    pca = PCA(n_components=n_comp, random_state=BASE_SEED)

    Xtr_pca_flat = scaler_pca.fit_transform(X_pca[tr_mask].reshape(-1, nf_pca))
    Xvl_pca_flat = scaler_pca.transform(X_pca[vl_mask].reshape(-1, nf_pca))
    Xte_pca_flat = scaler_pca.transform(Xte_pca.reshape(-1, nf_pca))

    Xtr_p = pca.fit_transform(Xtr_pca_flat).reshape(ntr, 25, n_comp)
    Xvl_p = pca.transform(Xvl_pca_flat).reshape(nvl, 25, n_comp)
    Xte_p = pca.transform(Xte_pca_flat).reshape(nte, 25, n_comp)

    Xtr = np.concatenate([Xtr_main, Xtr_p], axis=-1).astype(np.float32)
    Xvl = np.concatenate([Xvl_main, Xvl_p], axis=-1).astype(np.float32)
    Xte = np.concatenate([Xte_main_scaled, Xte_p], axis=-1).astype(np.float32)
    return Xtr, Xvl, Xte


class GRUBase(nn.Module):
    def __init__(self, d, h=128, nl=2, do=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)
        self.gru = nn.GRU(d, h, nl, batch_first=True, dropout=do, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(h * 2, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        b, t, f = x.shape
        x = self.bn(x.reshape(b * t, f)).reshape(b, t, f)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, c, dil, do=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
        self.bn1 = nn.BatchNorm1d(c)
        self.bn2 = nn.BatchNorm1d(c)
        self.do = nn.Dropout(do)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.c1(x)))
        x = self.do(x)
        x = F.relu(self.bn2(self.c2(x)))
        x = self.do(x)
        return x + r


class TCNModel(nn.Module):
    def __init__(self, d, h=96):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)
        self.proj = nn.Conv1d(d, h, 1)
        self.b1 = TCNBlock(h, 1)
        self.b2 = TCNBlock(h, 2)
        self.b4 = TCNBlock(h, 4)
        self.b8 = TCNBlock(h, 8)
        self.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        b, t, f = x.shape
        x = self.bn(x.reshape(b * t, f)).reshape(b, t, f).transpose(1, 2)
        x = self.proj(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b4(x)
        x = self.b8(x)
        x = x.transpose(1, 2)
        return self.head(x).squeeze(-1)


def inverse_log1p(p):
    return np.clip(np.expm1(p), 0, None)


def train_deep_sequence_1seed(
    model_fn, model_name, X_main, X_pca, Y_raw_seq, Xte_main, Xte_pca, scenario_folds,
    train_idx, test_idx, n_train_rows, n_test_rows, pca_n=0, seed=BASE_SEED,
):
    n_scen = X_main.shape[0]
    nte = Xte_main.shape[0]
    Y_log = np.log1p(Y_raw_seq).astype(np.float32)
    oof_seq = np.zeros((n_scen, 25), dtype=np.float32)
    tst_seq = np.zeros((nte, 25), dtype=np.float32)
    fold_maes = []

    for fi in range(N_SPLITS):
        tr_mask = scenario_folds != fi
        vl_mask = scenario_folds == fi
        Xtr, Xvl, Xte = make_fold_deep_features(
            X_main=X_main,
            X_pca=X_pca,
            Xte_main=Xte_main,
            Xte_pca=Xte_pca,
            tr_mask=tr_mask,
            vl_mask=vl_mask,
            pca_n=pca_n,
        )

        nf = Xtr.shape[-1]
        seed_everything_torch(seed + fi)
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Y_log[tr_mask])),
            batch_size=DEEP_BATCH,
            shuffle=True,
        )

        model = model_fn(nf).to(DEEP_DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=5)

        Xvl_t = torch.FloatTensor(Xvl).to(DEEP_DEVICE)
        best_mae = float("inf")
        best_state = None
        bad = 0

        for ep in range(DEEP_EPOCHS):
            model.train()
            for bx, by in loader:
                bx = bx.to(DEEP_DEVICE)
                by = by.to(DEEP_DEVICE)
                opt.zero_grad()
                pred = model(bx)
                loss = F.l1_loss(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                vp_t = model(Xvl_t)
                vp = inverse_log1p(vp_t.detach().cpu().numpy())
                vm = mean_absolute_error(Y_raw_seq[vl_mask].flatten(), vp.flatten())

            sch.step(vm)
            if vm < best_mae:
                best_mae = float(vm)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if bad >= DEEP_PATIENCE:
                break

        model.load_state_dict(best_state)
        model.to(DEEP_DEVICE)
        model.eval()

        with torch.no_grad():
            vp_t = model(Xvl_t)
            vp = inverse_log1p(vp_t.detach().cpu().numpy())
            oof_seq[vl_mask] = vp.astype(np.float32)

            Xte_t = torch.FloatTensor(Xte).to(DEEP_DEVICE)
            tp_t = model(Xte_t)
            tp = inverse_log1p(tp_t.detach().cpu().numpy())
            tst_seq += tp.astype(np.float32) / N_SPLITS

        fold_maes.append(best_mae)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    oof_row = np.zeros(n_train_rows, dtype=np.float32)
    test_row = np.zeros(n_test_rows, dtype=np.float32)
    oof_row[train_idx.flatten()] = oof_seq.flatten()
    test_row[test_idx.flatten()] = tst_seq.flatten()
    return np.clip(oof_row, 0, None), np.clip(test_row, 0, None), fold_maes


def train_deep_sequence_multiseed(
    model_fn, base_model_name, final_model_name, X_main, X_pca, Y_raw_seq, Xte_main,
    Xte_pca, scenario_folds, train_idx, test_idx, n_train_rows, n_test_rows,
    pca_n=0, seeds=None,
):
    if seeds is None:
        seeds = [BASE_SEED]

    oof_list = []
    tst_list = []
    seed_rows = []

    for sd in seeds:
        seed_label = f"{base_model_name}_seed{sd}"
        oof, tst, fold_maes = train_deep_sequence_1seed(
            model_fn=model_fn,
            model_name=seed_label,
            X_main=X_main,
            X_pca=X_pca,
            Y_raw_seq=Y_raw_seq,
            Xte_main=Xte_main,
            Xte_pca=Xte_pca,
            scenario_folds=scenario_folds,
            train_idx=train_idx,
            test_idx=test_idx,
            n_train_rows=n_train_rows,
            n_test_rows=n_test_rows,
            pca_n=pca_n,
            seed=sd,
        )

        b = mae_bucket_report(y_raw, oof)
        seed_rows.append({
            "model": base_model_name,
            "seed": sd,
            "oof_mae": b["MAE_all"],
            "MAE_0_10": b["MAE_0_10"],
            "MAE_10_50": b["MAE_10_50"],
            "MAE_50_plus": b["MAE_50_plus"],
            "fold_mae_mean": float(np.mean(fold_maes)),
            "fold_mae_std": float(np.std(fold_maes)),
        })

        oof_list.append(oof)
        tst_list.append(tst)

    oof_avg = np.mean(oof_list, axis=0)
    tst_avg = np.mean(tst_list, axis=0)
    return np.clip(oof_avg, 0, None), np.clip(tst_avg, 0, None), seed_rows


def main():
    global y_raw

    seed_everything(BASE_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([ENVIRONMENT]).to_csv(OUTPUT_DIR / "environment_versions.csv", index=False)

    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    layout = pd.read_csv(DATA_DIR / "layout_info.csv")

    train_stage1, bt = stage1_fe(train_raw, layout, train_raw)
    test_stage1, _ = stage1_fe(test_raw, layout, test_raw, bt)

    train_stage1["layout_type"] = train_stage1["layout_type"].astype("category")
    test_stage1["layout_type"] = test_stage1["layout_type"].astype("category")

    y_raw = train_stage1[TARGET].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(train_stage1, groups=train_stage1["scenario_id"]))

    lgb_fixed_fe_name = "LGB_QUEUE_V2_1SEED_SQRT"
    lgb_fixed_with_queue = True
    lgb_fixed_with_rel = False

    all_results = []
    all_single_transform_rows = []
    t0 = time.time()

    train_lgb_fixed, test_lgb_fixed, fixed_fc, fixed_build_info = build_lgb_stage2_dataset(
        train_stage1=train_stage1,
        test_stage1=test_stage1,
        y_raw=y_raw,
        folds=folds,
        with_queue=lgb_fixed_with_queue,
        with_rel=lgb_fixed_with_rel,
        seed=BASE_SEED,
        fe_name=lgb_fixed_fe_name,
    )

    oof_fixed, tst_fixed, fixed_info = train_lgb_sqrt_only(
        train=train_lgb_fixed,
        test=test_lgb_fixed,
        fc=fixed_fc,
        y_raw=y_raw,
        folds=folds,
        seed=BASE_SEED,
        label=lgb_fixed_fe_name,
    )

    fixed_extra = {
        "with_queue": lgb_fixed_with_queue,
        "with_rel": lgb_fixed_with_rel,
        "v15_auc": fixed_build_info["v15_auc"],
        "pass1_mae": fixed_build_info["pass1_mae"],
        "stage2_feature_count": fixed_build_info["stage2_feature_count"],
        "target_transform": "sqrt",
        "seconds": time.time() - t0,
    }

    lgb_fixed_results = []
    add_result_row(
        rows=lgb_fixed_results,
        exp_name=lgb_fixed_fe_name,
        model_type="LGB_SQRT_ONLY_FIXED_QUEUE_V2",
        seed_mode="1seed",
        fe_name=lgb_fixed_fe_name,
        oof_pred=oof_fixed,
        y_true=y_raw,
        extra=fixed_extra,
    )
    add_result_row(
        rows=all_results,
        exp_name=lgb_fixed_fe_name,
        model_type="LGB_SQRT_ONLY_FIXED_QUEUE_V2",
        seed_mode="1seed",
        fe_name=lgb_fixed_fe_name,
        oof_pred=oof_fixed,
        y_true=y_raw,
        extra=fixed_extra,
    )

    for r in fixed_info["single_transform_rows"]:
        r["fe_name"] = lgb_fixed_fe_name
        r["with_queue"] = lgb_fixed_with_queue
        r["with_rel"] = lgb_fixed_with_rel
        all_single_transform_rows.append(r)

    lgb_fixed_df = pd.DataFrame(lgb_fixed_results).sort_values("oof_mae").reset_index(drop=True)
    single_transform_df = pd.DataFrame(all_single_transform_rows)

    best_fe_name = lgb_fixed_fe_name
    best_train = train_lgb_fixed
    best_test = test_lgb_fixed
    best_fc = fixed_fc

    lgb_fixed_df.to_csv(OUTPUT_DIR / "v28_lgb_fixed_queue_v2_sqrt_result.csv", index=False)
    single_transform_df.to_csv(OUTPUT_DIR / "v28_lgb_single_sqrt_details.csv", index=False)

    seed_oofs = {}
    seed_tsts = {}
    seed_infos = {}
    seed_individual_rows = []

    for sd in MODEL_SEEDS:
        seed_label = f"{best_fe_name.replace('_1SEED_SQRT', '')}_seed{sd}_SQRT"
        if sd == BASE_SEED:
            oof = oof_fixed
            tst = tst_fixed
            info = {"target_transform": "sqrt", "oof_mae": mean_absolute_error(y_raw, oof)}
        else:
            oof, tst, info = train_lgb_sqrt_only(
                train=best_train,
                test=best_test,
                fc=best_fc,
                y_raw=y_raw,
                folds=folds,
                seed=sd,
                label=seed_label,
            )

        seed_oofs[sd] = oof
        seed_tsts[sd] = tst
        seed_infos[sd] = info
        b = mae_bucket_report(y_raw, oof)
        seed_individual_rows.append({
            "seed": sd,
            "fe_name": best_fe_name,
            "target_transform": "sqrt",
            "oof_mae": b["MAE_all"],
            "MAE_0_10": b["MAE_0_10"],
            "MAE_10_50": b["MAE_10_50"],
            "MAE_50_plus": b["MAE_50_plus"],
        })

    seed_individual_df = pd.DataFrame(seed_individual_rows).sort_values("oof_mae").reset_index(drop=True)
    seed_summary_rows = []
    seed_ensemble_oofs = {}
    seed_ensemble_tsts = {}

    for k in SEED_COUNTS:
        used = MODEL_SEEDS[:k]
        oof_avg = np.mean([seed_oofs[s] for s in used], axis=0)
        tst_avg = np.mean([seed_tsts[s] for s in used], axis=0)
        exp_name = f"{best_fe_name.replace('_1SEED_SQRT', '')}_{k}SEED_SQRT"
        seed_ensemble_oofs[exp_name] = oof_avg
        seed_ensemble_tsts[exp_name] = tst_avg
        b = mae_bucket_report(y_raw, oof_avg)
        seed_summary_rows.append({
            "exp_name": exp_name,
            "n_seed": k,
            "fe_name": best_fe_name,
            "target_transform": "sqrt",
            "oof_mae": b["MAE_all"],
            "MAE_0_10": b["MAE_0_10"],
            "MAE_10_50": b["MAE_10_50"],
            "MAE_50_plus": b["MAE_50_plus"],
            "used_seeds": str(used),
        })
        add_result_row(
            rows=all_results,
            exp_name=exp_name,
            model_type="LGB_SQRT_ONLY_FIXED_QUEUE_V2_SEED_AVG",
            seed_mode=f"{k}seed",
            fe_name=best_fe_name,
            oof_pred=oof_avg,
            y_true=y_raw,
            extra={"n_seed": k, "used_seeds": str(used), "target_transform": "sqrt"},
        )

    seed_summary_df = pd.DataFrame(seed_summary_rows)
    seed_summary_df["improve_vs_1seed"] = seed_summary_df["oof_mae"].iloc[0] - seed_summary_df["oof_mae"]
    seed_summary_df["delta_from_prev"] = seed_summary_df["oof_mae"].shift(1) - seed_summary_df["oof_mae"]
    seed_summary_df.to_csv(OUTPUT_DIR / "v28_lgb_seed_fixed_queue_v2_sqrt_results.csv", index=False)
    seed_individual_df.to_csv(OUTPUT_DIR / "v28_lgb_seed_individual_fixed_queue_v2_sqrt_results.csv", index=False)

    sid_to_fold = {}
    for fi, (_, vi) in enumerate(folds):
        for sid in train_stage1.iloc[vi]["scenario_id"].unique():
            sid_to_fold[sid] = fi

    ctx_model_name = f"CONTEXT_PCA_10_GRU_{MODEL_SEED_COUNT}SEED"
    train_ctx, test_ctx, fc_ctx, pca_cols_ctx = build_gru_feature_frame(train_stage1, test_stage1, context_pca_n=10)
    X_ctx, Xp_ctx, Y_ctx, Xte_ctx, Xtep_ctx, trs_ctx, tes_ctx, tridx_ctx, teidx_ctx = prepare_deep_sequence_arrays(
        train_ctx, test_ctx, fc_ctx, pca_cols_ctx,
    )
    sfa_ctx = np.array([sid_to_fold[s] for s in trs_ctx])
    oof_ctx_gru, tst_ctx_gru, ctx_seed_rows = train_deep_sequence_multiseed(
        model_fn=GRUBase,
        base_model_name="CONTEXT_PCA_10_GRU",
        final_model_name=ctx_model_name,
        X_main=X_ctx,
        X_pca=Xp_ctx,
        Y_raw_seq=Y_ctx,
        Xte_main=Xte_ctx,
        Xte_pca=Xtep_ctx,
        scenario_folds=sfa_ctx,
        train_idx=tridx_ctx,
        test_idx=teidx_ctx,
        n_train_rows=len(train_stage1),
        n_test_rows=len(test_stage1),
        pca_n=10,
        seeds=MODEL_SEEDS,
    )
    ctx_seed_df = pd.DataFrame(ctx_seed_rows)
    ctx_seed_df.to_csv(OUTPUT_DIR / f"v28_context_pca10_gru_{MODEL_SEED_COUNT}seed_details.csv", index=False)
    add_result_row(
        rows=all_results,
        exp_name=ctx_model_name,
        model_type="GRU_CONTEXT_PCA_10",
        seed_mode=f"{MODEL_SEED_COUNT}seed",
        fe_name="REL_QUEUE_V2_CONTEXT_PCA_10",
        oof_pred=oof_ctx_gru,
        y_true=y_raw,
        extra={
            "main_feature_count": len(fc_ctx),
            "pca_raw_feature_count": len(pca_cols_ctx),
            "pca_n": 10,
            "n_seed": MODEL_SEED_COUNT,
            "seed_detail_csv": f"v28_context_pca10_gru_{MODEL_SEED_COUNT}seed_details.csv",
        },
    )

    tcn_model_name = f"A6_TCN_BRANCH_{MODEL_SEED_COUNT}SEED"
    train_tcn, test_tcn, fc_tcn, pca_cols_tcn = build_gru_feature_frame(train_stage1, test_stage1, context_pca_n=0)
    X_tcn, Xp_tcn, Y_tcn, Xte_tcn, Xtep_tcn, trs_tcn, tes_tcn, tridx_tcn, teidx_tcn = prepare_deep_sequence_arrays(
        train_tcn, test_tcn, fc_tcn, pca_cols_tcn,
    )
    sfa_tcn = np.array([sid_to_fold[s] for s in trs_tcn])
    oof_a6_tcn, tst_a6_tcn, tcn_seed_rows = train_deep_sequence_multiseed(
        model_fn=TCNModel,
        base_model_name="A6_TCN_BRANCH",
        final_model_name=tcn_model_name,
        X_main=X_tcn,
        X_pca=Xp_tcn,
        Y_raw_seq=Y_tcn,
        Xte_main=Xte_tcn,
        Xte_pca=Xtep_tcn,
        scenario_folds=sfa_tcn,
        train_idx=tridx_tcn,
        test_idx=teidx_tcn,
        n_train_rows=len(train_stage1),
        n_test_rows=len(test_stage1),
        pca_n=0,
        seeds=MODEL_SEEDS,
    )
    tcn_seed_df = pd.DataFrame(tcn_seed_rows)
    tcn_seed_df.to_csv(OUTPUT_DIR / f"v28_a6_tcn_branch_{MODEL_SEED_COUNT}seed_details.csv", index=False)
    add_result_row(
        rows=all_results,
        exp_name=tcn_model_name,
        model_type="TCN_BRANCH",
        seed_mode=f"{MODEL_SEED_COUNT}seed",
        fe_name="REL_QUEUE_V2",
        oof_pred=oof_a6_tcn,
        y_true=y_raw,
        extra={
            "main_feature_count": len(fc_tcn),
            "pca_raw_feature_count": 0,
            "pca_n": 0,
            "n_seed": MODEL_SEED_COUNT,
            "seed_detail_csv": f"v28_a6_tcn_branch_{MODEL_SEED_COUNT}seed_details.csv",
        },
    )

    lgb_final_exp_name = seed_summary_df.loc[seed_summary_df["n_seed"].idxmax(), "exp_name"]
    final_oof_candidates = {
        lgb_final_exp_name: seed_ensemble_oofs[lgb_final_exp_name],
        ctx_model_name: oof_ctx_gru,
        tcn_model_name: oof_a6_tcn,
    }
    final_test_candidates = {
        lgb_final_exp_name: seed_ensemble_tsts[lgb_final_exp_name],
        ctx_model_name: tst_ctx_gru,
        tcn_model_name: tst_a6_tcn,
    }

    names, weights, best_mae, final_oof = scipy_weight_ensemble(
        final_oof_candidates,
        y_raw,
        n_trials=SCIPY_TRIALS,
        seed=BASE_SEED + 2028 + MODEL_SEED_COUNT,
    )
    tst_mat = np.column_stack([np.clip(final_test_candidates[n], 0, None) for n in names])
    final_test = np.clip(tst_mat @ weights, 0, None)

    weight_rows = []
    for n, w in zip(names, weights):
        single_mae = mean_absolute_error(y_raw, final_oof_candidates[n])
        weight_rows.append({"name": n, "weight": float(w), "single_oof_mae": float(single_mae)})

    final_model_name = f"V28_FINAL_SCIPY_{MODEL_SEED_COUNT}SEED_3MODEL"
    final_submission_name = f"submission_{final_model_name}.csv"
    submitted_submission_name = f"submission_V28_FINAL_{MODEL_SEED_COUNT}SEED_3MODEL.csv"
    add_result_row(
        rows=all_results,
        exp_name=final_model_name,
        model_type="SCIPY_WEIGHTED_ENSEMBLE",
        seed_mode=f"{MODEL_SEED_COUNT}seed_each_model",
        fe_name="LGB_SQRT_PLUS_CONTEXT_PCA10_GRU_PLUS_TCN",
        oof_pred=final_oof,
        y_true=y_raw,
        extra={
            "candidate_count": len(names),
            "weights": str(dict(zip(names, weights))),
            "scipy_oof_mae": best_mae,
            "lgb_final_exp_name": lgb_final_exp_name,
            "gru_final_exp_name": ctx_model_name,
            "tcn_final_exp_name": tcn_model_name,
        },
    )

    all_results_df = pd.DataFrame(all_results).sort_values("oof_mae").reset_index(drop=True)
    all_results_df.to_csv(OUTPUT_DIR / "v28_final_all_results.csv", index=False)
    pd.DataFrame(weight_rows).sort_values("weight", ascending=False).to_csv(OUTPUT_DIR / "v28_final_scipy_weights.csv", index=False)

    oof_out = pd.DataFrame({"ID": train_stage1["ID"].values, "y_true": y_raw})
    for name, pred in final_oof_candidates.items():
        oof_out[name] = np.clip(pred, 0, None)
    oof_out[final_model_name] = np.clip(final_oof, 0, None)
    oof_out.to_csv(OUTPUT_DIR / f"v28_final_oof_preds_{MODEL_SEED_COUNT}seed_3model.csv", index=False)

    test_out = pd.DataFrame({"ID": test_stage1["ID"].values})
    for name, pred in final_test_candidates.items():
        test_out[name] = np.clip(pred, 0, None)
    test_out[final_model_name] = np.clip(final_test, 0, None)
    test_out.to_csv(OUTPUT_DIR / f"v28_final_test_preds_{MODEL_SEED_COUNT}seed_3model.csv", index=False)

    for name, pred in final_test_candidates.items():
        safe_name = name.replace("/", "").replace(" ", "")
        sub = pd.DataFrame({"ID": test_stage1["ID"].values, TARGET: np.clip(pred, 0, None)})
        sub = sub.sort_values("ID").reset_index(drop=True)
        sub.to_csv(OUTPUT_DIR / f"submission_{safe_name}.csv", index=False)

    sub_final = pd.DataFrame({"ID": test_stage1["ID"].values, TARGET: np.clip(final_test, 0, None)})
    sub_final = sub_final.sort_values("ID").reset_index(drop=True)
    sub_final.to_csv(OUTPUT_DIR / final_submission_name, index=False)
    sub_final.to_csv(OUTPUT_DIR / submitted_submission_name, index=False)


if __name__ == "__main__":
    main()
