# Changelog

---

## v19 (2026-04-28)

### 변경
- **scenario_id label-encoding 제거**: 카테고리 raw ID 피처에서 제외 → `te__scenario_id`만으로 시나리오 정보 접근
  - 이유: test 시나리오 전부 unseen → raw ID는 train에만 존재하는 노이즈
- **scenario_id TE smoothing 강화**: 10 → 100 (train/test 분포 격차 최소화)
  - layout_id, layout_type, layout_cluster는 smoothing=10 유지

### 추가
- **Scenario percentile rank 피처** (lag 대신): 8개 컬럼의 시나리오 내 백분위 순위 (`*_scene_pct`)
  - 대상: `congestion_score`, `charging_ratio_raw`, `low_battery_ratio`, `order_inflow_15m`, `blocked_path_15m`, `near_collision_15m`, `avg_trip_distance`, `max_zone_density`
  - lag와 달리 ordering-independent → 정렬 순서 버그 없음
- **CatBoost 추가**: Optuna 40 trials, CPU, Bernoulli bootstrap, early stopping 100
  - catboost 미설치 시 자동 skip (try/except)
- **MLP 추가**: sklearn MLPRegressor (256, 128, 64), ReLU, Adam, max_iter=300
  - StandardScaler 정규화, early_stopping=True, GroupKFold CV
- **4-way 블렌드**: XGB / LGB / CAT / MLP 가중치 grid search (step=0.05)

### 제거
- **Lag / diff 피처** (v18): 순서 의존적 + Dacon 점수 악화 → 제거

### 버그 수정 (v19b)
- **scenario_id fallback encoding 버그**: `select_dtypes("object")` fallback 루프가 scenario_id를 재포함 → TE 이후 raw `scenario_id` 컬럼 자체를 drop으로 완전 차단
  - `encoded cat cols` 로그 확인: ['layout_id', 'layout_type', 'layout_cluster', 'scenario_id'] → 버그 존재 확인

### 출력 파일
- `*_v5.csv` → `*_v6.csv` (v19 초기) → `*_v7.csv` (scenario_id drop 수정 후)

---

## v18 (2026-04-26)

### 추가
- **SCENE_COLS 확장** (12→18): `battery_mean`, `avg_charge_wait`, `charge_efficiency_pct`, `aisle_traffic_score`, `path_optimization_score`, `intersection_wait_time_avg` 추가 → scene 피처 ~60→~90개
- **Lag / diff 피처**: 시나리오 내 이전 timeslot 값 (lag1, lag2) + 변화율 (diff1) × 6컬럼 = 18개 신규 피처
  - 대상: `congestion_score`, `charging_ratio_raw`, `low_battery_ratio`, `order_inflow_15m`, `blocked_path_15m`, `near_collision_15m`
  - 첫 timeslot NaN → scene mean으로 대체
- **Scene × availability_ratio 교차**: `scene_congestion_score_mean`, `scene_max_zone_density_mean`, `scene_near_collision_15m_mean`, `scene_avg_trip_distance_mean` × `availability_ratio_log1p` 4쌍

### 변경
- **Target Encoding**: 단순 mean → Bayesian smoothed TE (`smoothing=10`)
  - `(n × group_mean + 10 × global_mean) / (n + 10)` — 소수 샘플 카테고리 분산 억제

### 출력 파일
- `*_v4.csv` → `*_v5.csv`

---

## v17 (2026-04-26)

### 추가
- **SCENE_COLS 확장** (5→12): `fault_count_15m`, `near_collision_15m`, `blocked_path_15m`, `avg_trip_distance`, `max_zone_density`, `task_reassign_15m`, `avg_recovery_time` 추가 → scene 피처 ~25→~60개
- **Timeslot rank 피처**: 시나리오 내 위치 (0~24) — `timeslot_rank`, `timeslot_rank_norm`, `timeslot_sin`, `timeslot_cos`
- **availability_ratio 교차**: `avail_x_congestion`, `avail_x_order_pressure`, `avail_x_low_battery`
- **avg_trip_distance 교차**: `trip_x_congestion`, `trip_x_density`, `trip_x_order` (LGB 2위 피처 활용)
- **미사용 컬럼 interaction**: `path_opt_x_congestion`, `intersection_wait_x_count`, `charge_eff_x_charging`, `wms_x_order`, `agv_success_x_util`
- **Scene vs 현재 상태 interaction 확장** (2→6쌍): `order_vs_scene`, `utilization_vs_scene`, `battery_vs_scene`, `trip_vs_scene` 추가

### 변경
- `N_LAYOUT_CLUSTERS`: 15 → 20
- `N_TRIALS_XGB / N_TRIALS_LGB`: 60 → 80
- CUDA VMM 오류 대응: Optuna trial → `TrialPruned`, Final CV → CPU fallback

### 출력 파일
- `*_v3.csv` → `*_v4.csv`

---

## v16 (2026-04-25)

### 추가
- **Layout clustering**: 레이아웃 300개를 KMeans(k=15)로 클러스터링 → `layout_cluster` 피처 + TE
  - unseen test layout도 cluster 배정으로 TE 혜택 가능
- **Scenario context 피처**: 시나리오 내 25개 timeslot 전체 집계 (mean/std/max/min/range) × 5컬럼 = ~25개
  - 대상: `charging_ratio_raw`, `congestion_score`, `low_battery_ratio`, `order_inflow_15m`, `robot_utilization`
- **availability_ratio 다항식**: `availability_ratio_sq`, `availability_ratio_log1p`, `availability_ratio_inv`
- **Combo TE 추가**: `te__cluster_type` (layout_cluster × layout_type)
- **Scene vs 현재 interaction**: `charging_vs_scene_mean`, `charging_vs_scene_ratio`, `congestion_vs_scene_mean`
- `XGBOOST_DISABLE_VMM=1` 환경변수 설정

### 출력 파일
- `*_v3.csv` (유지)

---

## v15 (2026-04-24)

### 추가
- **LightGBM 앙상블 첫 적용**: XGB + LGB blend (XGB weight 0.64)
- **OOF blend 최적화**: XGB weight grid search (0.0~1.0, step 0.02)

### 출력 파일
- `*_v3.csv` 도입

---

## v14 (2026-04-21)

### 추가
- FE: `charging_robot_per_charger`, `queue_pressure`, `delay_risk_proxy` 추가 (123 features)

### 변경
- Optuna 탐색 범위 조정: `learning_rate` 0.015~0.05, `max_depth` 6~8
- Optuna 60trials로 증가

---

## v13 (2026-04-21)

- FE core_plus(23개) + conditional features, sqrt transform, OOF TE, 133 features

---

## v12 (2026-04-21)

- FE lite(10개) + conditional features, sqrt transform, OOF TE, 120 features

---

## v11 (2026-04-21)

- FE lite(10개) + conditional features(flags), log1p transform, OOF TE, 120 features

---

## v10 (2026-04-19)

### 추가
- **OOF Target Encoding 첫 적용**: `layout_id`, `scenario_id`, `layout_type`, `te__layout_scenario` combo TE
- FE core_plus(23개) + conditional features, 133 features

### 변경
- log1p target transform

---

## v9 (2026-04-16)

### 변경
- **sqrt target transform 도입** (log1p → sqrt)
- FE lite(116 features)

---

## v8 (2026-04-16)

### 추가
- FE lite + 8개 추가 (116→124 features)
- `sample_weight`: 고지연 샘플 가중치 부여 (이후 사용 안 함 — Dacon 하락)

---

## v7 (2026-04-15)

### 변경
- log1p target transform 도입 (이후 sqrt로 교체)

---

## v6 (2026-04-15)

- FE core_plus mode (lite + 3개), 116→119 features

---

## v5 (2026-04-15)

- FE lite mode (상위 10개만 유지), 129→116 features

---

## v4 (2026-04-14)

### 추가
- FE 추가: ratio / product / diff 23개, 106→129 features

---

## v1~v3 (2026-04-14)

- v1: XGB only, Optuna 20trials, GroupKFold(5) layout+scenario, 106 features, layout_info merge
- v2: Optuna 50trials, 탐색범위 narrowing → 성능 하락
- v3: TPESampler seed 고정, OOF clip(0, None)
