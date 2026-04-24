# 스마트 물류창고 출고 지연 예측 — Daicon 경진대회

AMR 기반 스마트 물류창고 운영 스냅샷 데이터로 **향후 30분간 평균 출고 지연 시간(분)** 을 예측하는 회귀 모델입니다.

**CV 베스트 (v15):** Blend CV MAE `9.0919` | **Dacon 베스트 (v14):** Public `10.6792`

---

## 접근 방법 요약

| 단계 | 내용 |
|------|------|
| **피처 엔지니어링** | 로봇 상태 비율, 혼잡도 교호작용, 배터리 리스크, 주문 압력, 순환 인코딩 등 108개 → 163개 파생 피처 |
| **타깃 변환** | `sqrt` 변환으로 학습 후 `square` 역변환 (MAE 안정화) |
| **OOF Target Encoding** | `layout_id`, `scenario_id`, `layout_type`, 조합 TE (리크 없이 GroupKFold 내 적용) |
| **검증 전략** | `GroupKFold(5)` — `layout_id + scenario_id` 기준 그룹 분할 |
| **하이퍼파라미터 튜닝** | Optuna TPE (XGBoost 60 trial, LightGBM 60 trial) |
| **앙상블** | XGBoost + LightGBM, OOF 기반 최적 가중치 그리드 서치 (XGB 0.64) |
| **평가 지표** | MAE |

---

## 파일 구조

```
smart-warehouse-delay-prediction/
├── data/
│   ├── train.csv                        # 학습 데이터 (250,000 × 94)
│   ├── test.csv                         # 평가 데이터 (50,000 × 93)
│   ├── layout_info.csv                  # 창고 레이아웃 보조 정보
│   └── sample_submission.csv            # 제출 양식
├── logs/                                # 버전별 실행 로그
├── daycon.py                            # 메인 학습 & 추론 스크립트
├── dockerfile                           # CUDA 12.8.1 기반 GPU 실행 환경
├── docker-compose.yml
├── requirements.txt
├── PLAN.md                              # 대회 전략 계획서
├── results.md                           # 버전별 실험 결과 추적
└── README.md
```

실행 후 출력 파일은 모두 `data/` 디렉터리에 저장됩니다.

```
data/
├── submission_ensemble_v3.csv           # 최종 제출 파일
├── oof_ensemble_v3.csv                  # OOF 진단 (layout별 오차 포함)
├── feature_importance_xgb_v3.csv        # XGBoost 피처 중요도
└── feature_importance_lgb_v3.csv        # LightGBM 피처 중요도
```

---

## 환경 설정 및 실행

### Docker (권장 — GPU 환경)

```bash
docker compose up
```

`data/` 경로는 `docker-compose.yml`의 볼륨 마운트로 지정합니다.  
기본값: `/home/plasma/test/data:/data` — 환경에 맞게 수정하세요.

### 직접 실행 (CPU 환경)

```bash
pip install -r requirements.txt
pip install lightgbm
python daycon.py
```

> GPU가 없으면 [daycon.py:256](daycon.py#L256) 에서 `device="cuda"` → `device="cpu"` 로 변경하세요.

**실행 시간:** Optuna 120 trial + 최종 CV × 2모델 × 5 fold 기준 GPU 약 1~2시간

**개발 환경**

| 항목 | 내용 |
|------|------|
| Python | 3.10+ |
| CUDA | 12.8.1 |
| XGBoost | GPU (`device="cuda"`, `tree_method="hist"`) |
| LightGBM | CUDA 빌드 (`CMAKE_ARGS="-DUSE_CUDA=1"`) |

---

## 파이프라인 상세

### 1. 데이터 로드 및 레이아웃 조인

```python
train = pd.read_csv("data/train.csv")
train = train.merge(layout_info, on="layout_id", how="left")
```

`layout_info.csv` 를 `layout_id` 기준으로 병합하여 창고 구조 정보를 피처에 추가합니다.

### 2. 피처 엔지니어링 (`make_features`)

| 카테고리 | 피처 예시 |
|----------|-----------|
| 로봇 상태 비율 | `charging_ratio`, `idle_ratio`, `active_ratio`, `availability_ratio` |
| 충전 비율 다항식 | `charging_ratio_sq`, `charging_ratio_cube`, `charging_ratio_log1p` |
| 혼잡도 교호작용 | `congestion_x_density`, `blocked_x_congestion`, `fault_x_congestion` |
| 주문 압력 | `order_per_active_robot`, `order_per_pack_station`, `pack_utilization` |
| 배터리 리스크 | `battery_risk_score`, `charging_pressure`, `queue_pressure` |
| 복합 지연 프록시 | `delay_risk_proxy` = `order_per_active_robot × (1+congestion) × (1+low_battery)` |
| 시간 인코딩 | `shift_hour_sin`, `shift_hour_cos` |

108개 원본 피처 → 163개 (FE + TE 포함)

### 3. 타깃 변환

```python
y_sqrt = sqrt(y_raw)   # 학습 시 사용
pred   = sqrt_pred ** 2  # 역변환 후 MAE 계산
```

### 4. OOF Target Encoding

`layout_id`, `scenario_id`, `layout_type`, `layout_id × scenario_id` 조합에 대해 GroupKFold 내에서 리크 없이 TE를 적용합니다.

### 5. Optuna 하이퍼파라미터 튜닝

- **XGBoost**: 60 trial, `reg:absoluteerror` objective, CUDA 가속
- **LightGBM**: 60 trial, `mae` objective

### 6. 최적 앙상블 가중치 탐색

OOF 예측값으로 XGB 가중치를 0 ~ 1 (0.02 간격) 그리드 서치하여 MAE 최소화 가중치를 선택합니다.

```python
final_pred = w_xgb * xgb_test + (1 - w_xgb) * lgb_test
# v15 기준: w_xgb = 0.64
```

---

## 주요 피처 (v15 기준)

**XGBoost 상위 피처** (gain 기준)

| 피처 | 중요도 비율 |
|------|------------|
| `availability_ratio` | 30.7% |
| `charging_ratio_log1p` | 17.3% |
| `charging_ratio_sq` | 12.2% |
| `charging_ratio_cube` | 11.9% |
| `charging_ratio` | 9.7% |

**LightGBM 상위 피처** (split 기준): `scenario_id`, `sku_concentration`, `te__layout_id`, `avg_trip_distance`, `zone_dispersion`

---

## 제출 파일 형식

```
ID,avg_delay_minutes_next_30m
TEST_00000,3.142
TEST_00001,1.876
...
```
