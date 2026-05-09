# 공부 정리 노트

---

## 목차

1. [데이터 구조](#1-데이터-구조)
2. [Feature Engineering](#2-feature-engineering)
3. [OOF Target Encoding](#3-oof-target-encoding)
4. [XGBoost 하이퍼파라미터](#4-xgboost-하이퍼파라미터)
5. [모델 학습 기법](#5-모델-학습-기법)
6. [1등 코드 분석](#6-1등-코드-분석)

---

## 1. 데이터 구조

### 계층 구조

```
레이아웃 (300개 창고)
  └── 시나리오 (여러 개)
        └── 타임슬롯 (25개, 15분 간격)
                └── 타깃: avg_delay_minutes_next_30m
```

- `train.csv` 250,000행 = 10,000 시나리오 × 25 타임슬롯
- `test.csv`  50,000행 = 2,000 시나리오 × 25 타임슬롯

---

### 컬럼 분류

#### 식별자
| 컬럼 | 설명 |
|---|---|
| `ID` | 행 고유 ID |
| `layout_id` | 어느 창고인지 (WH_001~WH_300) |
| `scenario_id` | 어느 시뮬레이션 시나리오인지 |

#### 레이아웃 구조 (`layout_info.csv`에서 merge)
창고 자체의 물리적 특성. **시간이 지나도 변하지 않는 정적 값.**

- 공간: `floor_area_sqm`, `ceiling_height_m`, `aisle_width_avg`
- 설비: `charger_count`, `pack_station_count`, `robot_total`
- 구조: `intersection_count`, `one_way_ratio`, `layout_compactness`, `zone_dispersion`
- 기타: `building_age_years`, `fire_sprinkler_count`, `emergency_exit_count`

#### 운영 상태 (`train.csv` 원본)
타임슬롯마다 측정되는 **실시간 변화값.**

- **로봇 상태**: `robot_active`, `robot_idle`, `robot_charging`, `robot_utilization`
- **혼잡/장애**: `congestion_score`, `blocked_path_15m`, `near_collision_15m`, `fault_count_15m`
- **배터리**: `low_battery_ratio`, `charge_queue_length`, `avg_charge_wait`
- **주문/물류**: `order_inflow_15m`, `unique_sku_15m`, `avg_trip_distance`
- **밀도**: `max_zone_density`, `task_reassign_15m`, `avg_recovery_time`

#### 타깃
| 컬럼 | 설명 |
|---|---|
| `avg_delay_minutes_next_30m` | 다음 30분간 평균 지연 시간 (예측 대상) |

---

## 2. Feature Engineering

### make_features — 피처 조합으로 새 피처 만들기

XGBoost/LightGBM 같은 트리 모델은 `A × B` 같은 곱셈/나눗셈 관계를 직접 학습하기 어렵다.  
직접 만들어줘야 모델이 그 상호작용을 쉽게 포착한다.

#### 사용하는 4가지 연산

**1. 비율 (ratio)**
```python
"charging_ratio" = robot_charging / robot_total
"idle_ratio"     = robot_idle / robot_total
```
절대값보다 비율이 레이아웃 크기와 무관하게 비교 가능.

**2. 곱 (product)**
```python
"congestion_x_density" = congestion_score × max_zone_density
"battery_risk_score"   = low_battery_ratio × robot_idle
```
두 신호가 **동시에 높을 때** 위험하다는 상호작용 표현.

**3. 다항식 (polynomial)**
```python
"charging_ratio_sq"    = charging_ratio²
"charging_ratio_cube"  = charging_ratio³
"charging_ratio_log1p" = log(1 + charging_ratio)
```
충전 비율이 높아질수록 지연이 급격히 늘어나는 비선형 관계를 잡기 위해.

**4. 시나리오 평균과의 차이**
```python
"charging_vs_scene_mean"  = charging_ratio - scene_charging_ratio_raw_mean
"congestion_vs_scene_mean" = congestion_score - scene_congestion_score_mean
```
"지금 이 타임슬롯이 이 시나리오 평균보다 얼마나 나쁜가" 표현.

---

### SCENE_COLS — 시나리오 문맥 요약

**선택 기준**: 운영 상태를 나타내는 지표 (타임슬롯마다 변하는 값들).

```python
SCENE_COLS = [
    "charging_ratio_raw",   # (파생) robot_charging / robot_total
    "congestion_score",     # (원본)
    "low_battery_ratio",    # (원본)
    "order_inflow_15m",     # (원본)
    "robot_utilization",    # (원본)
    "fault_count_15m",      # (원본)
    "near_collision_15m",   # (원본)
    "blocked_path_15m",     # (원본)
    "avg_trip_distance",    # (원본)
    "max_zone_density",     # (원본)
    "task_reassign_15m",    # (원본)
    "avg_recovery_time",    # (원본)
]
```

`make_scenario_context`에서 이 컬럼들을 시나리오 단위로 집계한다:
```
SCENE_COL 하나당 5개 파생 피처 생성
scene_congestion_score_mean  → 시나리오 평균
scene_congestion_score_std   → 시나리오 표준편차
scene_congestion_score_max   → 시나리오 최대값
scene_congestion_score_min   → 시나리오 최소값
scene_congestion_score_range → 시나리오 범위 (max - min)
```

> 타임슬롯 단위 값만으로는 전체 흐름을 모르니, 시나리오 전체를 요약해서 각 행에 붙여주는 것.

---

### make_layout_clusters — 창고 유형 군집화

**선택 기준**: 레이아웃의 물리적/구조적 특성 중 **숫자형(numeric)** 컬럼.

```python
num_cols = [
    "aisle_width_avg", "intersection_count", "one_way_ratio",
    "pack_station_count", "charger_count", "layout_compactness",
    "zone_dispersion", "robot_total", "building_age_years",
    "floor_area_sqm", "ceiling_height_m", "fire_sprinkler_count",
    "emergency_exit_count",
]
```

- 시간에 따라 **변하지 않는** 정적 값만 사용 (운영 지표 제외)
- KMeans는 거리 계산을 하므로 수치형만 가능

**목적**: 300개 레이아웃을 유사한 창고끼리 묶어 `layout_cluster` 피처 생성.  
test에만 있는 unseen 레이아웃도 클러스터에 배정 → Target Encoding 혜택 가능.

---

### add_timeslot_rank — 타임슬롯 위치 인코딩

```python
df["timeslot_rank"]      # 0~24 정수
df["timeslot_rank_norm"] # 0.0~1.0 정규화
df["timeslot_sin"]       # sin(2π × rank / 25)
df["timeslot_cos"]       # cos(2π × rank / 25)
```

**sin/cos의 의미**: 0~24를 원 위의 점으로 변환 → **순환성 표현**.

```
rank=0  → sin≈0.0,  cos=1.0   (시작)
rank=24 → sin≈-0.25, cos≈0.97  (시작 바로 직전)
```

rank=0과 rank=24가 sin/cos 값이 비슷 → 모델이 "둘이 가깝다"고 인식.

> **현실적으로는**: 이 데이터에서 25개 타임슬롯이 하루를 반복하는 구조가 아닐 수 있어서  
> 순환성 자체가 의미 없을 수 있다. sin/cos는 **시간이 명확히 순환할 때** (시각 0~23시, 요일 0~6) 진짜 효과적인 기법.

---

### EPS (Epsilon)

```python
EPS = 1e-6  # = 0.000001
```

**0으로 나누기 방지**용 아주 작은 수.

```python
def _r(n, d):
    return df[n] / (df[d] + EPS)  # d가 0이어도 안 터짐
```

- `0 / (0 + 0.000001)` = 0 → 안전하게 처리
- `5 / (10 + 0.000001)` ≈ 0.5 → 값이 거의 변하지 않음

수치 계산의 "안전망". 데이터 스케일에 비해 충분히 작으면 어떤 값이든 상관없다.

---

## 3. OOF Target Encoding

### Target Encoding이란?

범주형 컬럼의 각 카테고리 값을 **해당 카테고리의 타겟 평균값**으로 치환하는 인코딩 기법.

```
예시)
layout_id = "A" → 평균 지연시간 5.0분
layout_id = "B" → 평균 지연시간 2.3분

te__layout_id 컬럼:
  행이 layout_id=A이면 → 5.0
  행이 layout_id=B이면 → 2.3
```

트리 기반 모델(XGBoost, LightGBM)에서 Label Encoding보다 효과적인 경우가 많다.  
카테고리의 **서수 관계가 없어도** 타겟과의 관계를 수치로 잘 표현하기 때문.

---

### 문제: 데이터 누수 (Leakage)

학습 데이터 전체로 TE를 계산하면, 각 행의 타겟값이 **자기 자신의 피처 계산에 포함**된다.

```
train 전체로 layout_id=A의 평균 계산
  → 그 행 자신의 타겟도 평균에 포함됨
  → 모델이 "피처를 보면 타겟을 이미 알 수 있는" 상태가 됨
  → 학습 성능은 높지만 실전에서 급락 (과적합)
```

---

### 해결책: OOF (Out-Of-Fold) TE

Cross-Validation의 fold 구조를 활용해 **자기 자신의 타겟을 보지 않고** TE를 계산한다.

```
전체 데이터를 K개 fold로 분할

for each fold i:
    나머지 (K-1)개 fold → 인코더 학습
    fold i → 학습된 인코더로 변환 (자기 타겟 안 봄)

test 데이터:
    train 전체로 인코더 학습 후 변환
    (test는 타겟이 없으므로 leakage 없음)
```

```
Fold 1 [────────] train
Fold 2 [────────] train  →  인코더 학습
Fold 3 [────────] train
Fold 4 [────────] train
Fold 5 [  val   ]        →  학습된 인코더로 변환 (leakage 없음)
```

---

### Unseen 카테고리 처리

test에 학습 데이터에 없던 카테고리가 등장하면 TE 값을 알 수 없다.  
→ **전체 타겟 평균(global mean)** 으로 대체 (`fillna(gmean)`)

---

### 콤보 TE

두 카테고리를 조합해 더 세밀한 패턴을 캡처.

```
layout_id + scenario_id → "A__scenario_1" 조합의 평균 지연시간
layout_cluster + layout_type → "cluster_3__type_B" 조합의 평균 지연시간
```

단, 조합 수가 많을수록 각 조합의 샘플 수가 줄어드는 **sparsity 문제** 주의.

---

### 핵심 요약

| 항목 | 내용 |
|---|---|
| 목적 | 범주형 변수를 타겟 기반 수치로 변환 |
| 문제 | naive TE는 leakage 발생 |
| 해결 | OOF — fold별로 자기 타겟 제외하고 계산 |
| Unseen 처리 | global mean으로 fallback |
| 응용 | 콤보 TE로 더 세밀한 패턴 캡처 가능 |

---

## 4. XGBoost 하이퍼파라미터

### 트리 구조 관련

**`max_depth`** — 트리 최대 깊이

- 작다 → 단순한 규칙만 학습
- 크다 → 복잡한 조건 조합까지 학습, 과적합 위험

**`min_child_weight`** — 분기 후 자식 노드의 최소 데이터 무게

> "이 정도 데이터도 안 모이면 굳이 가지를 만들지 마."

- 작음 → 세세하게 쪼갬 → 과적합 위험
- 큼 → 충분한 데이터가 있을 때만 쪼갬 → 안정적

**`gamma`** — 분기 허용 손실 임계값

> "가지 하나 더 만들 거면, 성능 개선이 확실해야 해."

- `gamma = 0` → 손실이 조금만 줄어도 분기 허용
- `gamma` 큼 → 손실이 많이 줄어야만 분기 허용

---

### 샘플링 관련

**`subsample`** — 각 트리를 만들 때 전체 행 중 몇 %를 사용할지

```
subsample = 0.8 → 매 트리마다 데이터의 80%만 랜덤하게 사용
```

**`colsample_bytree`** — 트리당 전체 피처 중 몇 %를 사용할지

**`colsample_bylevel`** — 트리의 각 깊이(level)마다 피처를 다시 샘플링

```
colsample_bytree=0.8, colsample_bylevel=0.8
→ 각 level에서 실제 사용 피처 비율 ≈ 0.8 × 0.8 = 0.64
```

---

### 정규화 관련

**`reg_alpha`** (L1 정규화)
> "별로 중요하지 않은 리프 가중치는 아예 0에 가깝게 만들어."

피처가 많거나 노이즈가 많은 경우 유용. 희소 모델 만들기.

**`reg_lambda`** (L2 정규화)
> "각 리프의 예측값이 너무 커지지 않게 조심해."

전체 가중치를 전반적으로 억제.

| | reg_alpha | reg_lambda |
|---|---|---|
| 방식 | 필요 없는 건 끊어내기 | 전체적으로 힘을 줄이기 |
| 결과 | 희소 모델 | 전체 가중치 억제 |

---

### 학습 효율 관련

**`n_estimators`** — 부스팅 트리를 최대 몇 개까지 만들지

`early_stopping_rounds`와 함께 쓰면 성능이 더 이상 좋아지지 않을 때 자동으로 멈춤.  
→ 보통 넉넉히 크게 잡고 early stopping으로 최적 개수를 찾는 방식 사용.

**`max_bin`** — 연속형 피처를 몇 개 구간으로 나눌지 (`tree_method="hist"` 전용)

```
max_bin = 10  → 0~100 범위를 10개 구간으로 대충 나눔 (빠름)
max_bin = 512 → 512개 구간으로 정밀하게 나눔 (느림)
```

---

### 한눈에 요약

| 파라미터 | 역할 | 크면 |
|---|---|---|
| `max_depth` | 트리 최대 깊이 | 복잡한 패턴, 과적합 위험 |
| `min_child_weight` | 분기 최소 샘플 무게 | 단순한 트리, 안정적 |
| `gamma` | 분기 허용 손실 임계값 | 보수적 분기 |
| `subsample` | 행 샘플링 비율 | 안정적, 과적합 가능 |
| `colsample_bytree` | 트리당 피처 비율 | 많은 피처 사용 |
| `colsample_bylevel` | 깊이당 피처 비율 | 많은 피처 사용 |
| `reg_alpha` | L1 정규화 | 가중치 희소화 |
| `reg_lambda` | L2 정규화 | 가중치 전반 억제 |
| `n_estimators` | 최대 트리 수 | 더 많이 학습 (early stopping 권장) |
| `max_bin` | 히스토그램 구간 수 | 정밀하지만 느림 |

---

## 5. 모델 학습 기법

### Bootstrap — 샘플링 방식 (CatBoost)

트리를 만들 때 전체 학습 데이터를 다 쓰지 않고 **일부를 랜덤 샘플링**하는 기법.  
과적합을 줄이고 모델 다양성을 높이는 효과.

`bootstrap_type`은 그 샘플링 방식을 결정:

| 타입 | 방식 |
|---|---|
| **Bernoulli** | 각 샘플을 독립적으로 확률 p로 포함/제외 (XGBoost의 `subsample`과 동일) |
| Bayesian | 베이지안 방식으로 가중치 부여 (CatBoost 기본값) |
| MVS | 그래디언트 기준으로 중요한 샘플 우선 선택 |

코드에서 `Bernoulli`를 선택한 이유:  
→ **`subsample` 파라미터를 Optuna로 튜닝하려면 Bernoulli 타입이어야 하기 때문.**  
CatBoost 기본값인 Bayesian은 subsample을 지원하지 않음.

---

### Pseudo-labeling — 정답 없는 데이터 활용

**정답이 없는 test 데이터에 정답을 임시로 붙여서 학습에 활용하는 기법.**

```
[1단계] 일반 학습
train 250,000행 (정답 O)
        ↓ 학습
      모델
        ↓ 예측
test 50,000행 → 예측값 (정답 X)

[2단계] 예측값을 "가짜 정답(pseudo-label)"으로 사용
train 250,000행 (진짜 정답)
   +
test  50,000행 (가짜 정답)
= 300,000행
        ↓ 재학습
    최종 모델
        ↓ 최종 예측
```

**왜 효과가 있냐면**:  
train과 test의 분포가 다를 때, 모델이 재학습 시 test 분포도 함께 보게 되어 격차가 줄어든다.

**주의점**:
- 가짜 정답이라 1차 예측이 틀렸으면 그 오류가 재학습에 전파됨
- 1차 모델이 어느 정도 좋아야 효과 있음
- 반복할수록 효과가 수렴 (이 대회에서 2라운드부터 거의 개선 없음 확인)

---

### 시드 앙상블 (Seed Ensemble)

**같은 모델을 여러 개의 랜덤 시드로 학습한 후 예측을 평균내는 기법.**

```python
ENSEMBLE_SEEDS = [42, 123, 456]

# 각 시드로 학습 → 예측
predictions = [model(seed=s).predict(test) for s in ENSEMBLE_SEEDS]

# 평균
final_pred = np.mean(predictions, axis=0)
```

**효과**: 예측의 분산(variance)이 줄어들어 안정적인 성능.  
- 피처 변경이 없으므로 regression(성능 하락) 위험 없음
- 트리 모델에서 시드가 바뀌면 feature/row subsampling 순서가 달라져 다른 트리가 만들어짐

**한계**: 편향(bias)은 줄이지 못함. 근본적인 모델 개선이 아니라 분산 감소.  
이 대회에서 3-seed 앙상블으로 Dacon 10.3999 → 10.3987 (+0.0012) 개선 확인.

---

### CatBoost란?

Yandex가 만든 그래디언트 부스팅 라이브러리. XGBoost, LightGBM과 같은 계열.

**핵심 특징:**
- **범주형 변수 자동 처리**: 인코딩 없이 직접 넣을 수 있음
- **Ordered boosting**: 과적합 방지를 위한 독자적 학습 알고리즘
- **느리지만 정확**: LGB보다 학습 속도 느리지만 일반화 성능이 종종 더 좋음

**설치**: `pip install catboost`

---

## 6. 1등 코드 분석

> 파일: `docs/codeOfRank1.py`  
> 최종 성적: **1등 / 1022팀**, 우리: 256등 (MAE 10.3987)

---

### 전체 파이프라인 구조

```
[Stage 1 FE]
  stage1_fe()                   : 기본 피처 + lag/rolling/expanding
  add_queueing_features_v2()    : 대기행렬 이론 피처 (M/M/1 공식)
  add_scenario_relative_features() : 시나리오 내 z-score, percentile rank
        ↓
[Stage 2 Build] — build_lgb_stage2_dataset()
  Pass 1 LGB로 OOF 예측
  → 예측값 자체를 새 피처로 변환 (pred_lag1, cum_pred 등)
  → 시나리오 수준 분류기 (고지연 여부, 최대 지연 예측)
  → layout Target Encoding (OOF)
  → pseudo_delay proxy 피처
        ↓
[모델 3종 × 10 seed]
  LGB × 10 seed          : sqrt transform, early stopping
  GRU × 10 seed          : 시나리오를 25스텝 시계열로 처리
  TCN × 10 seed          : dilated convolution, 시계열 처리
        ↓
[최종 앙상블]
  scipy Nelder-Mead로 3개 모델의 최적 가중치 탐색 (80 random restart)
```

---

### 우리와의 핵심 차이

#### 설정 비교

| 항목 | 우리 (v27) | 1등 |
|---|---|---|
| 검증 전략 | 5-fold GroupKFold (layout+scenario) | **10-fold GroupKFold (scenario만)** |
| 트리 모델 | XGB + LGB | LGB만 |
| 하이퍼파라미터 | Optuna 60 trial | **고정값** (튜닝 없음) |
| LGB n_estimators | 9,500 (lr=0.005) | 5,000 (lr=0.03) |
| 딥러닝 | MLP (효과 없음) | **GRU + TCN (시계열 처리)** |
| 시드 수 | 3 seed | **10 seed** |
| 앙상블 가중치 | 그리드 서치 | **scipy Nelder-Mead** |
| 스태킹 | 없음 | **2단계 스태킹** |
| 노이즈 컬럼 제거 | 없음 | **25개 사전 제거** |

---

### 차이점 1 — 시나리오 내 시계열 피처

우리가 v18에서 시도했다가 버그로 포기했던 기법. 1등은 항상 `groupby("scenario_id")` 안에서 처리해서 안전하게 구현.

```python
g = df.groupby("scenario_id")

# lag 피처 — 직전 타임슬롯 값
df["congestion_score_lag1"]    = g["congestion_score"].shift(1)
df["order_inflow_15m_lag1"]    = g["order_inflow_15m"].shift(1)

# rolling mean — 최근 3개 타임슬롯 평균
df["congestion_score_rolling3"] = g["congestion_score"].transform(lambda x: x.rolling(3,1).mean())

# diff — 직전 대비 변화량
df["congestion_diff1"] = g["congestion_score"].diff(1)
```

**우리 코드의 버그 원인 추정**: `groupby` 없이 전체 df에 `shift(1)`을 적용하면, 다른 시나리오의 마지막 행이 현재 시나리오의 첫 행에 섞여들어감.

**expanding window — "지금까지 누적/최대" 포착:**

```python
df["max_congestion_so_far"] = g["congestion_score"].transform(lambda x: x.expanding().max())
df["cum_congestion"]        = g["congestion_score"].cumsum()
df["consec_congestion"]     = ...  # 연속 혼잡 횟수 (cc 함수로 구현)
df["cong_frac_so_far"]      = g["congestion_score"].transform(lambda x: (x > 0).expanding().mean())
df["scenario_progress"]     = df["timeslot"] / max_timeslot  # 진행률 0.0~1.0
```

이 피처들의 의미: 현재 타임슬롯이 시나리오의 어느 단계인지, 지금까지 상황이 얼마나 나빴는지를 담음.

---

### 차이점 2 — 대기행렬 이론 피처 (M/M/1 Queue)

수학적으로 검증된 대기행렬 이론 공식을 피처로 직접 구현.

```python
eps = 1e-6

# ρ (rho) = 부하율. ρ → 1에 가까울수록 대기 폭발
df["pack_rho"] = df["pack_utilization"].clip(0, 0.999)

# M/M/1 큐 — 폭발 지수 (ρ가 1에 가까울수록 지수적으로 커짐)
df["pack_explosion_log"] = np.log1p(1.0 / (1.0 - df["pack_rho"] + eps))

# M/M/1 큐 — 평균 대기 행렬 길이
df["pack_wait_mm1"] = df["pack_rho"]**2 / (1.0 - df["pack_rho"] + eps)

# Kingman 공식 (G/G/1) — 변동성(CV²)까지 반영한 대기시간
order_cv_squared = (order_std / order_mean)**2
df["kingman_wait_pack"] = (rho / (1 - rho)) * (1 + order_cv_squared) / 2
```

**핵심 개념**: ρ = 0.9 → 0.95로 오를 때 대기시간이 선형이 아니라 **지수적으로 폭발**함.  
우리의 `delay_risk_proxy = order × (1+congestion) × (1+battery)` 와 방향은 같지만, 실제 물리 공식 기반이라 더 정확함.

---

### 차이점 3 — 미래 context 피처

같은 시나리오의 **미래 행**을 참조하는 피처. leakage가 아닌 이유는, test 데이터도 25개 행 전체가 주어지기 때문.

```python
for c in cols:
    for w in [2, 3, 5]:
        # 현재 기준 과거 w개 평균
        df[f"ctx_{c}_past{w}_mean"]  = g[c].transform(lambda x: x.shift(1).rolling(w,1).mean())

        # 현재 기준 미래 w개 평균
        df[f"ctx_{c}_future{w}_mean"] = g[c].transform(lambda x: x.shift(-1).rolling(w,1).mean())

        # 미래 - 과거 → 트렌드 방향
        df[f"ctx_{c}_future{w}_minus_past{w}"] = future - past
```

우리 SCENE_COLS (시나리오 전체 평균/std)과 비슷한 발상이지만, 각 행마다 **현재 위치 기준의 국소 맥락**을 제공.

---

### 차이점 4 — 2단계 스태킹

Pass 1 모델의 예측값을 피처로 추가해 Pass 2에서 재학습하는 구조.

```
[Pass 1] LGB → OOF 예측값 생성
        ↓
[예측값 → 새 피처 변환]
  pred_lag1_log      : 직전 타임슬롯의 예측값 (log 변환)
  pred_rolling3      : 직전 3개의 예측값 평균
  pred_diff1         : 예측값 변화량
  cum_pred           : 예측값 누적합
  max_pred_so_far    : 지금까지 최대 예측값

[시나리오 수준 분류기] — 별도 LGB 모델들
  pred_p_high        : 이 시나리오가 고지연(>50분) 시나리오일 확률
  pred_mean_delay    : 시나리오 평균 지연 예측
  pred_max_delay     : 시나리오 최대 지연 예측
  pred_pct90         : 시나리오 90th percentile 지연 예측

[Pass 2] 위 피처들 전부 포함해서 최종 LGB 재학습
```

**효과**: "이 시나리오는 전체적으로 나쁜 시나리오인가?"라는 시나리오 수준 정보가 각 행에 추가됨.

---

### 차이점 5 — 딥러닝: 시나리오를 시계열로 처리

우리가 MLP를 써서 효과가 없었던 이유가 여기서 드러남.  
MLP는 각 행을 독립적으로 봄. GRU/TCN은 25개 타임슬롯을 **하나의 시퀀스**로 처리.

```python
# 핵심: 각 시나리오 25개 행을 (시나리오, 25스텝, 피처수) 3D 배열로 reshape
X = features.reshape(n_scenarios, 25, n_features)  # (10000, 25, 피처수)

# Bidirectional GRU — 앞뒤로 흐름 읽기
class GRUBase(nn.Module):
    def __init__(self, d, h=128):
        self.gru = nn.GRU(d, h, num_layers=2,
                          batch_first=True,
                          dropout=0.3,
                          bidirectional=True)   # 양방향
        self.head = nn.Linear(h*2, 1)          # 각 스텝별 예측
    # 출력: (batch, 25, 1) — 25개 타임슬롯 각각에 대한 예측

# TCN — dilated convolution으로 시계열 패턴 학습
# dilation 1→2→4→8 : receptive field가 지수적으로 확장됨 (25 타임슬롯 전체 커버)
```

**GRU가 학습할 수 있는 것들 (MLP는 불가)**:
- 혼잡이 N 타임슬롯 연속으로 지속되면 지연이 급증하는 패턴
- 시나리오 시작이 안정적이어도 중반 이후 터지는 패턴
- 회복 중인지 악화 중인지 방향성

---

### 노이즈 컬럼 25개 제거

SHAP 분석으로 무의미하다고 판단된 컬럼을 사전에 제거. 우리는 이 컬럼들을 전부 그대로 사용했음.

```python
NOISE_COLS = [
    # 기상 환경 — 창고 내부 지연과 무관
    "warehouse_temp_avg", "humidity_pct", "external_temp_c",
    "wind_speed_kmh", "precipitation_mm",
    # 건물 환경
    "lighting_level_lux", "ambient_noise_db", "floor_vibration_idx",
    "air_quality_idx", "co2_level_ppm", "hvac_power_kw",
    # 인프라/IT
    "wms_response_time_ms", "scanner_error_rate",
    "wifi_signal_db", "network_latency_ms",
    # 인사/안전
    "worker_avg_tenure_months", "safety_score_monthly",
    "return_order_ratio", "label_print_queue",
    "barcode_read_success_rate", "ups_battery_pct",
    "lighting_zone_variance", "robot_calibration_score",
    "racking_height_avg_m", "shift_handover_delay_min",
]
```

---

### 핵심 요약 — 우리가 더 잘할 수 있었던 것

| 기법 | 우리 시도 | 1등 | 교훈 |
|---|---|---|---|
| 시나리오 내 lag/rolling | 버그로 실패 (v18) | 정상 구현 | groupby 안에서 처리하면 안전 |
| 미래 context 피처 | 시도 안 함 | 핵심 피처로 활용 | 같은 시나리오 내 미래값은 leakage 아님 |
| M/M/1 대기행렬 피처 | 시도 안 함 | 핵심 피처 | 물리 공식 기반 피처가 강력 |
| 딥러닝 | MLP (행별 독립) | GRU/TCN (시퀀스) | 시나리오 단위 시계열로 봐야 함 |
| 2단계 스태킹 | 시도 안 함 | OOF 예측값을 피처로 재활용 | 예측값 자체가 유용한 피처 |
| 노이즈 제거 | 없음 | 25개 사전 제거 | 무관한 피처는 처음부터 제거 |
| 시드 수 | 3 seed | 10 seed | 시드 많을수록 분산 감소 |
