# 스마트 물류창고 출고 지연 예측

**Dacon 경진대회** — AMR(자율이동로봇) 기반 스마트 물류창고의 운영 스냅샷 데이터를 기반으로,  
향후 30분간 평균 출고 지연 시간(분)을 예측하는 회귀 모델입니다.

---

## 최종 성적

| 항목 | 값 |
|---|---|
| **순위** | **256 / 1022** (상위 25%) |
| **Public MAE** | **10.3987** |
| **CV MAE** | 8.8631 |
| **총 실험 버전** | v1 ~ v27 |
| **최종 피처 수** | 253개 |

> MAE 기준, 점수가 낮을수록 좋음

---

## 핵심 기법

| 기법 | 효과 |
|---|---|
| **시나리오 컨텍스트 피처** (SCENE_COLS × 5통계) | v16에서 Dacon 10.807 → 10.490 대도약 |
| **레이아웃 클러스터링** (KMeans k=20) | 미등장 layout에도 Target Encoding 적용 가능 |
| **OOF Target Encoding** | CV leakage 없이 범주형 변수 인코딩 |
| **sqrt 타깃 변환** | log1p 대비 CV-Dacon gap 안정화 |
| **XGB + LGB 앙상블** (0.10 : 0.90) | 단일 모델 대비 분산 감소 |
| **Pseudo-labeling** (1라운드) | 학습/테스트 분포 격차 일부 해소 |
| **3-seed 앙상블** | 예측 분산 추가 감소, 최종 +0.001 개선 |

---

## 성능 개선 여정

```
v1  : 10.807  XGBoost 단독, 기본 피처 106개
v9  : 10.686  sqrt transform 도입 (log1p 대비 gap 안정)
v14 : 10.679  피처 엔지니어링 최적화, XGB 단독 최고
v16 : 10.490  시나리오 컨텍스트 피처 + 레이아웃 클러스터링 ★ 대도약
v17 : 10.406  SCENE_COLS 12개 + interaction 피처 → 핵심 피처셋 완성
v23 : 10.400  Pseudo-labeling 도입
v27 : 10.399  3-seed 앙상블 → 최종 최고점 ★
```

---

## 파이프라인

```
[데이터 로드]
  train.csv + layout_info.csv (merge on layout_id)
        │
        ▼
[피처 엔지니어링]
  ① 기본 FE: 비율, 교호작용, 다항식, 주문 압력, 배터리 리스크 등
  ② 시나리오 컨텍스트: SCENE_COLS 12개 × {mean, std, max, min, range} = ~60개
  ③ 레이아웃 클러스터: KMeans(k=20) → layout_cluster 생성
  ④ 타임슬롯 인코딩: rank, norm, sin, cos
  ⑤ OOF Target Encoding: layout_id, scenario_id, layout_type, 조합 TE 6개
        │
        ▼
[sqrt 타깃 변환]   y_train = sqrt(delay_minutes)
        │
        ▼
[Optuna 튜닝]   XGBoost 60 trial / LightGBM 60 trial (GroupKFold 5)
        │
        ▼
[최종 CV]   3 seeds × 5 folds × 2 모델 = 30개 모델
  XGB OOF 평균 → xgb_oof / xgb_test
  LGB OOF 평균 → lgb_oof / lgb_test
        │
        ▼
[OOF 최적 블렌딩]   XGB 0.10 : LGB 0.90
        │
        ▼
[Pseudo-labeling]   blend_test → 테스트 pseudo-label → train+test(300k)으로 재학습
        │
        ▼
[최종 예측]   inv_sqrt → 제출
```

---

## 피처 구성 (253개)

| 피처 그룹 | 개수 | 설명 |
|---|---|---|
| 원본 컬럼 | ~108 | train.csv + layout_info merge |
| 비율 피처 | ~15 | charging_ratio, idle_ratio, active_ratio, availability_ratio 등 |
| 다항식 피처 | ~9 | charging_ratio_sq, _cube, 혼잡도 제곱 등 |
| 교호작용 피처 | ~12 | congestion×density, blocked×congestion, fault×order 등 |
| 주문 압력 피처 | ~8 | order_per_active_robot, pack_utilization 등 |
| 배터리 리스크 피처 | ~6 | battery_risk_score, charging_pressure 등 |
| 복합 지연 프록시 | ~4 | delay_risk_proxy = order × (1+congestion) × (1+low_battery) 등 |
| 시나리오 컨텍스트 | ~60 | SCENE_COLS 12개 × {mean, std, max, min, range} |
| 타임슬롯 위치 | 4 | slot_rank, slot_norm, slot_sin, slot_cos |
| OOF Target Encoding | 6 | layout_id, scenario_id, layout_type, layout_cluster, 조합 2개 |

> 상세 내용: [docs/features.md](docs/features.md)

---

## 모델 구성

```python
# XGBoost (최종 하이퍼파라미터)
n_estimators  = 3,500
learning_rate = 0.0051
max_depth     = 9
device        = "cuda"

# LightGBM (최종 하이퍼파라미터)
n_estimators  = 9,500
learning_rate = 0.0055
num_leaves    = 2^11 = 2048

# 앙상블
final_pred = 0.10 * xgb_pred + 0.90 * lgb_pred

# 3-seed 앙상블 (seeds: 42, 123, 456)
xgb_pred = mean([xgb_seed42, xgb_seed123, xgb_seed456])
lgb_pred = mean([lgb_seed42, lgb_seed123, lgb_seed456])
```

**검증 전략:** `GroupKFold(n_splits=5)` — `layout_id + scenario_id` 기준 그룹 분할

---

## 핵심 인사이트

**CV-Dacon Gap (~1.54)의 원인**

테스트 데이터의 시나리오(scenario_id)가 학습 데이터와 **완전히 다름 (unseen scenarios)**.  
LightGBM의 1위 피처인 `scenario_id`(importance ~9000)가 테스트에서는 무용지물.  
GroupKFold는 validation fold에서 scenario_id가 노출되어 CV가 낙관적으로 편향됨.

```
→ 피처 추가 시 CV↑ but Dacon↓ 패턴이 반복되는 근본 원인
→ train에만 유효한 신호(scenario 연관 피처)를 추가할수록 gap이 확대됨
→ SCENE_COLS을 18개로 늘렸을 때(v18), CV 최고 but Dacon 역효과
```

**해결하지 못한 문제**
- gap 1.54는 단순 피처/하이퍼파라미터 튜닝으로 해소 어려운 구조적 문제
- WH_051, WH_073, WH_217 등 고오차 레이아웃 (~33점) 개선 실패

---

## 파일 구조

```
smart-warehouse-delay-prediction/
├── data/
│   ├── train.csv                         # 학습 데이터 (250,000 × 94)
│   ├── test.csv                          # 평가 데이터 (50,000 × 93)
│   ├── layout_info.csv                   # 창고 레이아웃 보조 정보
│   ├── sample_submission.csv             # 제출 양식
│   ├── submission_ensemble_v15.csv       # 최종 제출 파일
│   ├── oof_ensemble_v15.csv              # OOF 예측 (layout별 오차 포함)
│   ├── feature_importance_xgb_v15.csv    # XGBoost 피처 중요도
│   └── feature_importance_lgb_v15.csv    # LightGBM 피처 중요도
├── docs/
│   ├── features.md                       # 피처 전체 설명
│   ├── results.md                        # 버전별 실험 결과
│   ├── changelog.md                      # 버전별 변경 이력
│   └── study_notes.md                    # 학습 노트
├── logs/                                 # 버전별 실행 로그
├── daycon.py                             # 메인 학습 & 추론 스크립트
├── dockerfile                            # CUDA 12.8.1 기반 GPU 환경
├── docker-compose.yml
└── requirements.txt
```

---

## 실행 환경 및 방법

### Docker (권장 — GPU 환경)

```bash
docker compose up
```

`docker-compose.yml`의 볼륨 마운트 경로를 환경에 맞게 수정하세요.  
기본값: `/home/plasma/test/data:/data`

### 직접 실행

```bash
pip install -r requirements.txt
pip install lightgbm
python daycon.py
```

GPU가 없으면 `daycon.py` 내 `device="cuda"` → `device="cpu"` 로 변경하세요.

**실행 시간:** Optuna 120 trial + 최종 CV (30개 모델) 기준 GPU 약 2~3시간

### 개발 환경

| 항목 | 내용 |
|---|---|
| Python | 3.10+ |
| CUDA | 12.8.1 |
| XGBoost | GPU (`device="cuda"`, `tree_method="hist"`) |
| LightGBM | CUDA 빌드 (`CMAKE_ARGS="-DUSE_CUDA=1"`) |
| Optuna | 60 trials per model |

---

## 실험 로그

버전별 상세 실험 기록: [docs/results.md](docs/results.md)  
버전별 변경 이력: [docs/changelog.md](docs/changelog.md)

---

## 1등 코드 비교 분석

> 1등 코드: `docs/codeOfRank1.py` 참고

### 전체 구조 비교

| 항목 | 이 코드 (256등) | 1등 |
|---|---|---|
| 검증 전략 | 5-fold GroupKFold (layout+scenario) | 10-fold GroupKFold (scenario만) |
| 트리 모델 | XGBoost + LightGBM | LightGBM만 |
| 하이퍼파라미터 | Optuna 60 trial | 고정값 (튜닝 없음) |
| 딥러닝 | MLP (효과 없음) | GRU + TCN (시계열 처리) |
| 스태킹 | 없음 | 2단계 스태킹 |
| 앙상블 시드 수 | 3 seed | 10 seed |
| 앙상블 가중치 | 그리드 서치 | scipy Nelder-Mead |
| 노이즈 컬럼 제거 | 없음 | SHAP 기반 25개 제거 |

---

### 놓쳤던 핵심 기법 3가지

#### 1. 시나리오 내 시계열 피처

각 타임슬롯의 값을 독립적으로 보는 것이 아니라, **시나리오 안에서의 흐름**을 피처로 만드는 방식.

```python
g = df.groupby("scenario_id")

# 직전 타임슬롯 대비 변화
df["congestion_lag1"]    = g["congestion_score"].shift(1)
df["congestion_diff1"]   = g["congestion_score"].diff(1)
df["congestion_rolling3"] = g["congestion_score"].transform(lambda x: x.rolling(3,1).mean())

# 지금까지의 누적/최대 — "얼마나 나쁜 상황이 쌓였는가"
df["max_congestion_so_far"] = g["congestion_score"].transform(lambda x: x.expanding().max())
df["cum_congestion"]        = g["congestion_score"].cumsum()
df["scenario_progress"]     = df["timeslot"] / max_timeslot  # 0.0 ~ 1.0
```

v18에서 시도했으나 `groupby` 없이 전체 df에 `shift(1)`을 적용해 시나리오 경계가 섞이는 버그 발생. 정상 구현 시 효과가 있었을 가능성이 높음.

#### 2. 대기행렬 이론 피처 (M/M/1 Queue)

패킹 스테이션/로봇/충전기의 가동률(ρ)을 물리 공식으로 변환.  
ρ → 1에 가까울수록 대기시간이 **지수적으로 폭발**하는 비선형 관계를 수식으로 표현.

```python
rho = pack_utilization.clip(0, 0.999)

# 폭발 지수 — rho=0.9 → 2.4, rho=0.99 → 4.6 (지수 증가)
pack_explosion_log = np.log1p(1.0 / (1.0 - rho))

# M/M/1 평균 대기 행렬 길이
pack_wait_mm1 = rho**2 / (1.0 - rho)

# Kingman 공식 — 주문 변동성(CV²)까지 반영
kingman_wait = (rho / (1 - rho)) * (1 + order_cv_squared) / 2
```

우리의 `delay_risk_proxy`와 방향은 같지만, 실제 물리 공식을 쓰면 모델이 비선형 관계를 더 쉽게 학습함.

#### 3. GRU/TCN — 시나리오를 시계열로 처리

MLP는 각 행을 독립적으로 처리. GRU/TCN은 시나리오의 25개 타임슬롯을 **하나의 시퀀스**로 읽음.

```python
# 25개 타임슬롯을 3D 시퀀스로 변환
X = features.reshape(n_scenarios, 25, n_features)  # (10000, 25, 피처수)

# Bidirectional GRU — 앞뒤 흐름을 동시에 학습
gru = nn.GRU(input_size, hidden=128, num_layers=2, bidirectional=True)
# 출력: 각 타임슬롯별 예측값 (25개)
```

GRU가 학습할 수 있는 패턴:
- 혼잡이 N 타임슬롯 연속으로 지속되면 지연이 급증
- 시나리오 초반이 안정적이어도 중반 이후 급격히 악화되는 패턴
- "회복 중"인지 "악화 중"인지 방향성

---

### 시도했으나 1등과 다른 선택

| 항목 | 우리 선택 | 이유 | 결과 |
|---|---|---|---|
| XGBoost 병행 사용 | Optuna로 튜닝 | XGB가 초반엔 LGB보다 강했음 | v17 이후 LGB가 더 강해짐, XGB 비중 0.10으로 감소 |
| SCENE_COLS 12개 | 18개에서 다시 줄임 | 18개는 과적합 | 핵심 판단이었음 — 1등도 소수 핵심 피처 선택 |
| Pseudo-labeling | 1라운드 적용 | 분포 격차 해소 목적 | 1등은 사용하지 않았음 |

---

### 회고 — 다음에 한다면

1. **lag/rolling 피처를 처음부터 제대로 구현** — `groupby("scenario_id")` 안에서 처리
2. **딥러닝은 행 단위가 아니라 시나리오 단위 시퀀스로** — GRU/TCN 구조 사용
3. **SHAP 분석으로 노이즈 컬럼 조기 제거** — 기상/건물 환경 컬럼 25개
4. **M/M/1 대기행렬 공식 피처 추가** — 가동률의 비선형 위험도 표현
5. **시드 수 늘리기** — 3 seed → 10 seed로 분산 추가 감소
