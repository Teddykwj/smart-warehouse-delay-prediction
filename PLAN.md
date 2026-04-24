# 스마트 물류창고 출고 지연 예측 AI 경진대회 — 계획서

## 1. 대회 개요

| 항목 | 내용 |
|------|------|
| **주제** | AMR 기반 스마트 물류창고 운영 데이터로 향후 30분간 평균 출고 지연 시간 예측 |
| **목표 변수** | `avg_delay_minutes_next_30m` (분 단위 연속형 회귀) |
| **평가 지표** | MAE (Mean Absolute Error) |
| **Public Score** | 전체 테스트 데이터 중 30% 샘플링 |
| **Private Score** | 전체 테스트 데이터 중 나머지 70% |
| **제출 언어** | Python |
| **일일 최대 제출** | 5회 |

---

## 2. 일정

| 날짜 | 내용 |
|------|------|
| 2026-04-01 | 대회 시작 |
| 2026-04-27 23:59 | 팀 병합 마감 |
| 2026-05-04 10:00 | **대회 종료 (최종 제출 마감)** |
| 2026-05-07 23:59 | 코드 & PPT 제출 마감 |
| 2026-05-08 ~ 05-15 | 코드 검증 |
| 2026-05-18 10:00 | 최종 순위 발표 |

---

## 3. 데이터 구조

```
open.zip
├── train.csv             # 학습 데이터 (250,000행 × 94컬럼)
├── test.csv              # 평가 데이터 (50,000행 × 93컬럼, 타깃 없음)
├── layout_info.csv       # 창고 레이아웃 보조 정보 (300행 × 15컬럼)
└── sample_submission.csv # 제출 양식 (50,000행 × 2컬럼)
```

### 컬럼 설명

| 파일 | 주요 컬럼 |
|------|-----------|
| `train.csv` | `ID`, `layout_id`, `scenario_id`, 90개 피처 컬럼, **`avg_delay_minutes_next_30m`** (타깃) |
| `test.csv` | `ID`, `layout_id`, `scenario_id`, 90개 피처 컬럼 |
| `layout_info.csv` | `layout_id`, 14개 레이아웃 관련 컬럼 |
| `sample_submission.csv` | `ID`, `avg_delay_minutes_next_30m` |

### 데이터 생성 방식
- 12,000개의 독립적인 창고 운영 시나리오 시뮬레이션
- 각 시나리오: 약 6시간 (25개 타임슬롯, 15분 간격)
- 피처 종류: 주문 정보, 로봇 이동, 배터리/충전, 혼잡도, 패킹 병목 등

---

## 4. 접근 전략

### 4-1. EDA (탐색적 데이터 분석)

- [x] 타깃 분포 확인 (왜도, 이상치) — `mean=18.96`, `max=715.86`, 오른쪽 꼬리 분포 → sqrt 변환 채택
- [x] 시나리오별 / 레이아웃별 지연 패턴 분석 — layout MAE 편차 큼 (WH_051 등 고오차 레이아웃 확인)
- [x] `layout_info.csv` 조인 후 레이아웃 특성과 지연 관계 분석
- [x] 피처 간 상관관계 및 중요도 분석 — `charging_ratio` 계열이 압도적 1위
- [ ] 시계열 패턴 분석 (타임슬롯 내 지연 추이) — 미실시

### 4-2. 피처 엔지니어링

- [x] **로봇 상태 비율**: `idle_ratio`, `charging_ratio`, `active_ratio`, `availability_ratio`
- [x] **충전 비율 다항식**: `charging_ratio_sq`, `_cube`, `_log1p` (지배적 피처 강화)
- [x] **혼잡도 교호작용**: `congestion_x_density`, `blocked_x_congestion`, `fault_x_congestion` 등
- [x] **주문 압력**: `order_per_active_robot`, `order_per_pack_station`, `pack_utilization`
- [x] **배터리/충전 리스크**: `battery_risk_score`, `charging_pressure`, `queue_pressure`
- [x] **복합 지연 프록시**: `delay_risk_proxy = order_per_active_robot × (1+congestion) × (1+low_battery)`
- [x] **레이아웃 조인**: `layout_id` 기준 `layout_info.csv` 병합
- [x] **시간 인코딩**: `shift_hour_sin`, `shift_hour_cos` (순환 인코딩)
- [x] **SKU 집중도**: `sku_concentration = order_inflow / unique_sku`
- [ ] **시계열 래그 피처**: 이전 슬롯 대비 변화량 — 미실시

### 4-3. 모델링

#### 구현된 앙상블

| 모델 | CV MAE (v15) | 역할 |
|------|-------------|------|
| XGBoost | 9.0950 | CUDA GPU, `reg:absoluteerror` |
| LightGBM | 9.1006 | CUDA 빌드, `mae` objective |
| **Blend (XGB 0.64)** | **9.0919** | OOF 기반 최적 가중치 |

#### 미실시 (초기 계획)

- CatBoost — 범주형 자동 처리 이점이 크지 않을 것으로 판단, 생략
- MLP / TabNet — 트리 모델 대비 성능 이점 불확실, 생략

#### 검증 전략

- **GroupKFold(5)**: `layout_id + scenario_id` 기준 그룹 분할 → 동일 시나리오의 타임슬롯이 train/val에 함께 들어가지 않도록 방지
- 각 fold에서 MAE 측정 후 평균

### 4-4. 하이퍼파라미터 튜닝

- **Optuna TPE** (seed=42 고정)
- XGBoost: 60 trial — `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha/lambda`, `gamma`, `max_bin`
- LightGBM: 60 trial — `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha/lambda`, `min_split_gain`

---

## 5. 파일 구조 (실제)

```
smart-warehouse-delay-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── layout_info.csv
│   ├── sample_submission.csv
│   └── (실행 후 생성) submission_ensemble_v3.csv 등
├── logs/                    # v1_log.txt ~ v15_log.txt
├── daycon.py                # 메인 스크립트 (단일 파일)
├── dockerfile
├── docker-compose.yml
├── requirements.txt
├── PLAN.md
├── results.md
└── README.md
```

---

## 6. 평가 지표 구현

```python
import numpy as np

def MAE(true, pred):
    return np.mean(np.abs(true - pred))
```

---

## 7. 제출 규칙 요약

- **허용 모델**: 공개 가중치 + MIT / Apache 2.0 / CC BY / CC BY-NC 라이선스
- **금지**: OpenAI API, Gemini API 등 외부 서버 의존 모델
- **외부 데이터**: 사용 가능 (단, test 데이터 학습 불가)
- **코드 요건**: UTF-8 인코딩, 오류 없이 실행, 입출력 경로 상대경로

---

## 8. 체크리스트

- [x] 데이터 다운로드 및 경로 확인
- [x] EDA 완료 (타깃 분포, 피처 분포, 결측값)
- [x] 레이아웃 보조 테이블 조인 실험
- [x] GroupKFold 기반 교차 검증 파이프라인 구축
- [x] LightGBM 베이스라인 제출
- [x] 피처 엔지니어링 반영 후 성능 비교
- [x] 앙상블 실험 (XGB + LGB)
- [x] 최종 제출 파일 형식 확인 (`ID`, `avg_delay_minutes_next_30m`)
- [ ] 코드 정리 및 README 작성 (2차 평가 대비)
- [ ] PPT 작성
