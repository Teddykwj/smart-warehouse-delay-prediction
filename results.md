# Experiment Results

> CV 베스트: **v15** — Blend CV MAE `9.0919`
>
> Dacon 베스트: **v14** — Dacon Public `10.6792` (CV MAE `9.1003`)
>
> CV-Dacon Gap 추이: 초기 ~1.59 → 현재 ~1.60 (안정적)

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

## 다음 실험 계획

| Version | 주요 변경 | 예상 효과 |
|---------|----------|----------|
| v16 | `pack_utilization` 버그 수정, `sku_concentration` 신규 피처, LGB GPU 빌드 적용, Optuna 탐색 범위 확장 | CV/Dacon gap 축소 기대 |

---

## 관찰 및 인사이트

- **`charging_ratio` 계열이 XGB 중요도의 ~62%** — 로봇 충전 비율이 지연 예측의 핵심 신호
- **log1p vs sqrt transform**: log1p는 CV 개선에도 Dacon gap 확대 경향 → sqrt 고정
- **sample_weight(고지연 강조)**: CV 최저 달성했으나 Dacon 하락 → 사용 안 함
- **FE 추가 시 주의**: 피처 수 증가가 항상 Dacon 개선으로 이어지지 않음 (v6, v8 참고)
- **LGB 앙상블 효과**: v15에서 XGB 단독 대비 CV MAE 0.003 개선 (9.0950 → 9.0919)
- **고오차 레이아웃**: WH_051(35.1), WH_217(34.5), WH_073(34.4) — 특정 레이아웃 전용 피처 필요 가능성
