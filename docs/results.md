# 실험 결과 최종 정리

## 최종 성적

| 항목 | 값 |
|---|---|
| **등수** | 256 / 1022 |
| **Dacon Public 최고점** | **10.3987** (v27) |
| **CV 최고점** | 8.8563 (v26, but Dacon은 악화) |
| **최적 CV** | 8.8631 (v27) |
| **최종 CV-Dacon Gap** | 1.536 |
| **총 실험 버전** | v1 ~ v27 |
| **최종 피처 수** | 253개 (TE 포함) |

---

## 성능 흐름

### 단계별 주요 전환점

```
v1  (10.807) → XGB 단독 시작
v9  (10.686) → sqrt transform 도입 (log1p 대비 gap 안정)
v14 (10.679) → FE 최적화, XGB 단독 최고점
v15 (10.691) → LGB 앙상블 첫 적용 (CV 개선, Dacon은 미달)
v16 (10.490) → 시나리오 컨텍스트 피처 + 레이아웃 클러스터링 → 대규모 도약
v17 (10.406) → SCENE_COLS 12개, interaction 피처 확장 → 핵심 피처셋 완성
v22 (10.434) → Pseudo-labeling 첫 도입
v23 (10.400) → v17 피처셋 정확 복귀 + pseudo-labeling → 당시 최고
v24 (10.400) → pseudo 2라운드 (수렴 확인, 사실상 동일)
v27 (10.399) → 3-seed 앙상블 → 최종 최고점
```

---

## 전체 버전 기록

| # | 버전 | Blend CV | Dacon | Gap | 핵심 변경 |
|---|---|---|---|---|---|
| 1 | v1 | - | 10.807 | 1.590 | XGB only, 106 features |
| 2 | v4 | 9.122 | 10.702 | 1.581 | FE 추가 (23개) |
| 3 | v9 | 9.105 | 10.686 | 1.581 | sqrt transform 도입 |
| 4 | v14 | 9.100 | 10.679 | 1.579 | XGB 단독 최고, FE 최적화 |
| 5 | v15 | 9.092 | 10.691 | 1.599 | XGB+LGB 앙상블 첫 적용 |
| 6 | **v16** | 8.943 | 10.490 | 1.547 | 시나리오 컨텍스트 + 레이아웃 클러스터 → **대도약** |
| 7 | **v17** | 8.864 | 10.406 | 1.542 | SCENE_COLS 12개, interaction 피처, → **핵심 피처셋** |
| 8 | v18 | 8.850 | 10.495 | 1.645 | SCENE_COLS 18개, lag/diff, Bayesian TE → 역효과 |
| 9 | v19 | 8.853 | 10.470 | 1.617 | scenario_id 격리 시도 → 버그로 실패 |
| 10 | v20 | 8.901 | 10.462 | 1.561 | layout_id만 GroupKFold, scenario_id drop |
| 11 | v21 | 8.862 | 10.474 | 1.612 | v17 인코딩 복귀 + SCENE_COLS 18 유지 |
| 12 | v22 | 8.869 | 10.434 | 1.565 | **Pseudo-labeling 첫 도입** |
| 13 | **v23** | 8.864 | 10.400 | 1.536 | v17 피처셋 복귀 + pseudo → **최고점 갱신** |
| 14 | v24 | 8.864 | 10.400 | 1.536 | pseudo 2라운드 (수렴, 개선 0.0000175) |
| 15 | v25 | 8.887 | 10.419 | 1.532 | layout context 추가 + scenario_id drop → 역효과 |
| 16 | v26 | 8.856 | 10.407 | 1.550 | lead + svl + layout context → CV 신기록, Dacon 악화 |
| 17 | **v27** ★ | 8.863 | **10.399** | 1.536 | **3-seed 앙상블** → **최종 최고점** |

---

## 최종 모델 구성 (v27)

### 피처셋 (253개)
| 피처 그룹 | 개수 | 설명 |
|---|---|---|
| 원본 컬럼 | ~108 | train.csv + layout_info merge |
| FE 파생 피처 | ~85 | 비율/곱/다항식/interaction |
| 시나리오 컨텍스트 | ~60 | SCENE_COLS 12개 × 5통계 |
| 타임슬롯 위치 | 4 | rank, norm, sin, cos |
| OOF TE | 6 | layout_id, scenario_id, layout_type, layout_cluster, layout_scenario combo, cluster_type combo |

### 모델 구성
```
XGBoost × 3 seeds (42, 123, 456) × 5 folds = 15개 모델
LightGBM × 3 seeds (42, 123, 456) × 5 folds = 15개 모델
→ 예측 평균 → 최적 블렌드 (XGB 0.10 : LGB 0.90)
→ Pseudo-labeling 재학습 (300k)
→ 최종 예측
```

### 주요 하이퍼파라미터 (v27 Optuna 결과)
| | XGBoost | LightGBM |
|---|---|---|
| n_estimators | 3,500 | 9,500 |
| learning_rate | 0.0051 | 0.0055 |
| max_depth | 9 | 11 |

---

## 효과 있었던 것

| 기법 | 효과 | 비고 |
|---|---|---|
| **sqrt transform** | gap 안정화 | log1p 대비 Dacon gap 작음 |
| **시나리오 컨텍스트 피처** | v16에서 Dacon 10.807→10.490 대도약 | SCENE_COLS 12개가 핵심, 18개는 과적합 |
| **레이아웃 클러스터링** | unseen layout에도 TE 가능 | KMeans(k=20) |
| **OOF Target Encoding** | naive TE 대비 leakage 없음 | simple mean이 Bayesian보다 효과적 |
| **XGB+LGB 앙상블** | 단일 모델 대비 안정적 | v17부터 LGB가 XGB보다 강해짐 |
| **Pseudo-labeling** | Dacon 10.406→10.400 (+0.006) | 1라운드에서 수렴, 2라운드 효과 없음 |
| **3-seed 앙상블** | Dacon 10.400→10.399 (+0.001) | 분산 감소, regression 위험 없음 |

---

## 효과 없었던 것 (교훈)

| 기법 | 결과 | 원인 |
|---|---|---|
| **log1p transform** | CV↑ Dacon↓ | gap 확대 |
| **sample_weight (고지연 강조)** | CV 최저 but Dacon 하락 | — |
| **SCENE_COLS 18개** | CV↑ Dacon↓ | 6개 추가분이 과적합 |
| **Bayesian TE (smoothing)** | Dacon 악화 | 오히려 정보 손실 |
| **lag 피처** | Dacon 악화 | 순서 의존적, 버그 발생 |
| **scenario percentile rank** | CV↑ Dacon↓ | CV 전용 신호 |
| **MLP 추가** | CV 10.13 (트리 대비 훨씬 낮음) | 이 데이터에서 기여 미미 |
| **scenario_id 제거** | CV +0.023 패널티 | 제거 비용 > 이득 |
| **layout context + lead + svl** | CV 신기록 but Dacon 악화 | layout_mean 계산 시 CV leakage 내재 |
| **pseudo 2라운드** | 개선 0.0000175 | 1라운드에서 이미 수렴 |

---

## 핵심 인사이트

**1. CV-Dacon Gap (~1.54)의 근본 원인**
- test 시나리오가 train과 완전히 다름 (unseen scenarios)
- scenario_id가 LGB 1위 피처(importance ~9000)이지만 test에서는 무용지물
- GroupKFold(layout+scenario)로 val에서 scenario가 노출되어 CV가 낙관적

**2. 피처 추가 패턴**
- 대부분의 피처 추가 → CV↑ but Dacon↓
- CV가 개선됐다고 Dacon도 개선되지 않음
- train에서만 유효한 신호(scenario 연관 피처)가 문제

**3. 가장 효과적인 조합**
```
v17 피처셋 (SCENE_COLS 12 + interaction) ← 핵심 피처
+ OOF TE (simple mean)                   ← Bayesian보다 효과적
+ Pseudo-labeling 1라운드                 ← 분포 격차 일부 해소
+ 3-seed 앙상블                           ← 분산 감소
= 최종 Dacon 10.3987
```

**4. 끝내 해결하지 못한 문제**
- CV-Dacon gap 1.54 → 구조적 문제, 단순 피처/하이퍼파라미터 튜닝으로 해소 어려움
- WH_051, WH_073, WH_217, WH_049 등 고오차 레이아웃 (~33점) 개선 실패
- 9.x대 진입을 위해서는 gap 자체를 줄이는 다른 접근 필요 (도메인 지식 기반 피처, 다른 CV 전략 등)
