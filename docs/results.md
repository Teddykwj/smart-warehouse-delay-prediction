# Experiment Results

> CV 베스트: **v18** — Blend CV MAE `8.8503`
>
> Dacon 베스트: **v17** — Dacon Public `10.4058` (CV MAE `8.8640`)
>
> CV-Dacon Gap 추이: 초기 ~1.59 → v15 ~1.60 → v16 1.547 → v17 1.542 → v18 1.645 → v19 1.617 → v20 1.561

| # | Date | Version | XGB CV MAE | LGB CV MAE | Blend CV MAE | XGB Weight | Dacon MAE | CV-Dacon Gap | Notes |
|---|------|---------|-----------|-----------|-------------|-----------|----------|-------------|-------|
| 1 | 2026-04-14 | v1 | 9.216826 | - | - | - | 10.806989 | 1.590163 | XGB only, Optuna 20trials, GroupKFold(5) layout+scenario, 106 features, layout_info merge |
| 2 | 2026-04-14 | v2 | - | - | - | - | 11.952655 | - | XGB only, Optuna 50trials, v1 best 기준 탐색범위 narrowing, reg:absoluteerror fixed, 성능 하락 |
| 3 | 2026-04-14 | v3 | - | - | - | - | 10.780596 | - | XGB only, TPESampler seed 고정, OOF clip(0,None), 탐색범위 동일, v1 대비 개선 |
| 4 | 2026-04-14 | v4 | 9.121760 | - | - | - | 10.702473 | 1.580713 | XGB only, FE추가(ratio/product/diff 23개), 106→129 features, Optuna 50trials |
| 5 | 2026-04-15 | v5 | 9.119698 | - | - | - | 10.690561 | 1.570863 | XGB only, FE lite mode(상위 10개만 유지), 129→116 features, Optuna 50trials |
| 6 | 2026-04-15 | v6 | 9.119474 | - | - | - | 10.708147 | 1.588673 | XGB only, FE core_plus mode(lite+3개), 116→119 features, CV는 개선됐으나 Dacon 하락 |
| 7 | 2026-04-15 | v7 | 9.102161 | - | - | - | 10.736758 | 1.634597 | XGB only, FE lite, log1p target transform, CV 개선됐으나 Dacon 하락, gap 확대 |
| 8 | 2026-04-16 | v8 | 9.091133 | - | - | - | 10.708259 | 1.617126 | XGB only, FE lite+8개 추가(116→124), sample_weight(고지연 가중치), CV 최고이나 Dacon 하락 |
| 9 | 2026-04-16 | v9 | 9.104861 | - | - | - | 10.686199 | 1.581338 | XGB only, FE lite(116), sqrt target transform, **베스트 갱신 (당시)** |
| 10 | 2026-04-19 | v10 | 9.097088 | - | - | - | 10.725590 | 1.628502 | XGB only, FE core_plus(23개)+conditional features, log1p transform, OOF TE, 133 features, Optuna 40trials |
| 11 | 2026-04-21 | v11 | 9.097470 | - | - | - | 10.692458 | 1.594988 | XGB only, FE lite(10개)+conditional features(flags), log1p transform, OOF TE, 120 features, Optuna 40trials |
| 12 | 2026-04-21 | v12 | 9.097254 | - | - | - | - | - | XGB only, FE lite(10개)+conditional features(flags), sqrt transform, OOF TE, 120 features, Optuna 40trials |
| 13 | 2026-04-21 | v13 | 9.100357 | - | - | - | - | - | XGB only, FE core_plus(23개)+conditional features, sqrt transform, OOF TE, 133 features, Optuna 40trials |
| 14 | 2026-04-21 | **v14** ★ | 9.100279 | - | - | - | **10.679176** | 1.578897 | XGB only, FE lite+tail(13개: +charging_robot_per_charger/queue_pressure/delay_risk_proxy), sqrt transform, OOF TE, 123 features, Optuna 50trials, 탐색범위 변경(lr 0.015-0.05, depth 6-8) — **Dacon 베스트** |
| 15 | 2026-04-24 | **v15** ★ | **9.094993** | **9.100564** | **9.091912** | **0.64** | 10.691098 | 1.599186 | **XGB+LGB 앙상블 첫 적용**, sqrt target, OOF TE, GroupKFold, 163 features — **CV 베스트** |

---

| 16 | 2026-04-25 | **v16** ★ | 8.947343 | 8.953853 | 8.942603 | 0.60 | **10.489869** | **1.547266** | 시나리오 컨텍스트 피처(25개), 레이아웃 클러스터링 K-means(15) + cluster TE, availability_ratio 다항식, 192→198 features, Optuna 60trials — **Dacon 베스트** |
| 17 | 2026-04-26 | **v17** ★ | 8.877776 | 8.865889 | 8.864023 | 0.26 | **10.405795** | **1.541772** | SCENE_COLS 5→12개, timeslot rank 피처, availability/trip/미사용컬럼 interaction 추가, scene×현재 interaction 2→6쌍, N_LAYOUT_CLUSTERS 15→20, Optuna 60→80trials — **Dacon 베스트** |
| 18 | 2026-04-26 | v18 | 8.858958 | 8.856323 | 8.850300 | 0.46 | 10.495149 | 1.645149 | SCENE_COLS 12→18개, lag/diff 피처 18개(→저중요도+Dacon악화), scene×avail_ratio 교차 4쌍, Smoothed TE, 305 features — lag 버그 수정 후 재제출했으나 v17 대비 Dacon 하락 |
| 19 | 2026-04-28 | v19 | 8.869424 | 8.869756 | 8.852557 | 0.45 | 10.470034 | 1.617477 | scenario_id TE smoothing 10→100, scenario percentile rank(8개), MLP 추가(CV 10.13, 기여 0.05), CatBoost 미설치(skip) — **버그**: fallback loop가 scenario_id 재포함, TE 약화(smoothing=100)+raw ID 혼재 → 양쪽 신호 모두 약해짐, CV·Dacon 모두 v17 대비 하락 |
| 20 | 2026-04-28 | v20 | 8.908573 | 8.913100 | 8.901196 | 0.56 | 10.462152 | 1.560956 | **GroupKFold by layout_id only**, scenario_id 완전 drop, MLP 제거, Trials 80→40 — v19(10.470)보다 소폭 개선됐으나 v17 베스트(10.406) 미달, scenario_id 제거 단독 효과는 긍정적이지만 충분하지 않음 |

---

## 관찰 및 인사이트

- **`charging_ratio` 계열이 XGB 중요도의 ~62%** — 로봇 충전 비율이 지연 예측의 핵심 신호
- **log1p vs sqrt transform**: log1p는 CV 개선에도 Dacon gap 확대 경향 → sqrt 고정
- **sample_weight(고지연 강조)**: CV 최저 달성했으나 Dacon 하락 → 사용 안 함
- **FE 추가 시 주의**: 피처 수 증가가 항상 Dacon 개선으로 이어지지 않음 (v6, v8 참고)
- **LGB 앙상블 효과**: v15에서 XGB 단독 대비 CV MAE 0.003 개선 (9.0950 → 9.0919)
- **고오차 레이아웃**: WH_073(33.4), WH_051(33.4), WH_217(32.6), WH_049(32.4), WH_098(29.8) — v17에서 전반적으로 소폭 개선 (v16 대비 WH_051: 34.6→33.4, WH_073: 33.9→33.4)
- **v17 XGB weight 급감**: 0.60→0.26 — LGB가 v17에서 XGB보다 강해짐 (LGB 8.8659 < XGB 8.8778)
- **v18 XGB weight 회복**: 0.26→0.46 — lag/diff 피처로 XGB-LGB 균형 회복
- **v18 LGB top 피처**: `scene_charge_efficiency_pct_*`, `scene_path_optimization_score_*` — v18 신규 SCENE_COLS가 즉시 상위권 진입
- **고오차 레이아웃 v18**: WH_051(33.5), WH_073(33.4), WH_217(32.9), WH_049(32.2), WH_098(29.7) — v17 대비 소폭 혼조
- **v19 scenario_id 격리 실패**: `cat_cols` 명시 목록에서 제거해도 fallback `select_dtypes("object")` 루프가 재포함 → TE smoothing 100으로 약화 + raw ID는 여전히 존재 → 양쪽 신호 모두 손상, gap 1.617로 오히려 악화
- **scenario_id 격리 교훈**: TE 이후 raw 컬럼을 `drop()`으로 완전 제거해야 의도한 격리 달성 (v7에 수정)
- **MLP 한계**: sklearn MLPRegressor CV 10.13 — 트리 모델 대비 현저히 약함, 이 데이터셋에서 앙상블 기여 미미 (0.05 weight)
