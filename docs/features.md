# 피처 정리

원본 데이터에 없는, 코드에서 새로 만든 피처들 설명.

---

## 피처 그룹 요약

| 그룹 | 개수 | 도입 버전 | 최종 포함 |
|---|---|---|---|
| Layout Clustering | 1 | v16 | ✅ |
| Scenario Context | ~60 | v16 | ✅ |
| Timeslot 위치 | 4 | v17 | ✅ |
| Robot State 비율 | 6 | v1~ | ✅ |
| charging_ratio 다항식 | 3 | v1~ | ✅ |
| availability_ratio 다항식 | 3 | v16 | ✅ |
| NaN 지시자 | 9 | v1~ | ✅ |
| 혼잡·밀도 상호작용 | 8 | v1~ | ✅ |
| 주문 처리량 | 8 | v1~ | ✅ |
| 배터리·충전 | 10 | v1~ | ✅ |
| 레이아웃·이동 | 4 | v1~ | ✅ |
| availability_ratio 상호작용 | 3 | v17 | ✅ |
| avg_trip_distance 상호작용 | 3 | v17 | ✅ |
| 미사용 컬럼 상호작용 | 5 | v17 | ✅ |
| 로봇 수 차이·합 | 2 | v1~ | ✅ |
| 복합 지연 위험 지수 | 2 | v14 | ✅ |
| 교대 시간 사이클 인코딩 | 3 | v1~ | ✅ |
| Scene × 현재 상태 상호작용 | 6 | v16~v17 | ✅ |
| OOF Target Encoding | 6 | v10 | ✅ |
| **합계** | **~253** | | |

---

## 최종 포함 피처

### 1. Layout Clustering

레이아웃의 물리적 특성으로 KMeans 클러스터링 (k=20).

| 피처 | 설명 |
|---|---|
| `layout_cluster` | 비슷한 구조의 레이아웃 그룹 (0~19) |

> unseen test 레이아웃도 클러스터에 배정 → TE 혜택 가능

---

### 2. Scenario Context Features

각 시나리오의 25개 타임슬롯 전체를 집계해 "이 시나리오 전반의 상황"을 요약.  
SCENE_COLS 12개 × 5통계 = **~60개** 피처 생성.

| 집계 대상 컬럼 | 의도 |
|---|---|
| `charging_ratio_raw` | 시나리오 전반의 충전 부하 수준 |
| `congestion_score` | 전반적인 혼잡 수준 |
| `low_battery_ratio` | 배터리 상황이 얼마나 나빴는지 |
| `order_inflow_15m` | 전반적인 수요 압력 |
| `robot_utilization` | 로봇 부하 수준 |
| `fault_count_15m` | 장애 발생 빈도 |
| `near_collision_15m` | 충돌 위험 수준 |
| `blocked_path_15m` | 경로 차단 빈도 |
| `avg_trip_distance` | 이동 거리 수준 |
| `max_zone_density` | 구역 밀집 수준 |
| `task_reassign_15m` | 작업 재배정 빈도 |
| `avg_recovery_time` | 회복 시간 수준 |

각 컬럼당 생성 통계: `mean` / `std` / `max` / `min` / `range`  
예: `scene_congestion_score_mean`, `scene_congestion_score_std`, ...

---

### 3. Timeslot 위치 피처

시나리오 내 25개 타임슬롯 중 몇 번째인지 표현.

| 피처 | 설명 |
|---|---|
| `timeslot_rank` | 순서 (0~24 정수) |
| `timeslot_rank_norm` | 정규화 순서 (0.0~1.0) |
| `timeslot_sin` | sin(2π × rank / 25) |
| `timeslot_cos` | cos(2π × rank / 25) |

---

### 4. Robot State 비율

로봇 상태를 전체 로봇 수 대비 비율로 변환.

| 피처 | 수식 | 의도 |
|---|---|---|
| `idle_ratio` | robot_idle / robot_total | 대기 중 로봇 비율 |
| `charging_ratio` | robot_charging / robot_total | 충전 중 로봇 비율 (**핵심 피처**) |
| `active_ratio` | robot_active / robot_total | 작업 중 로봇 비율 |
| `utilization_gap` | robot_idle / robot_active | 대기 대비 작업 비율 |
| `robot_available` | robot_active + robot_idle | 즉시 투입 가능 로봇 수 |
| `availability_ratio` | robot_available / robot_total | 가용 로봇 비율 |

---

### 5. charging_ratio 다항식

충전 비율이 지연에 **비선형적** 영향을 주므로 다항식 확장.

| 피처 | 수식 |
|---|---|
| `charging_ratio_sq` | charging_ratio² |
| `charging_ratio_cube` | charging_ratio³ |
| `charging_ratio_log1p` | log(1 + charging_ratio) |

---

### 6. availability_ratio 다항식

가용 로봇 비율의 비선형 관계 표현.

| 피처 | 수식 |
|---|---|
| `availability_ratio_sq` | availability_ratio² |
| `availability_ratio_log1p` | log(1 + availability_ratio) |
| `availability_ratio_inv` | 1 / availability_ratio |

---

### 7. NaN 지시자

결측 자체가 운영 이상 신호일 수 있어 별도 피처로 추가.

| 피처 | 의미 |
|---|---|
| `congestion_score_nan` | 혼잡도 측정 불가 |
| `fault_count_15m_nan` | 장애 집계 누락 |
| `near_collision_15m_nan` | 충돌 위험 집계 누락 |
| `avg_recovery_time_nan` | 회복 시간 측정 불가 |
| `charge_queue_length_nan` | 충전 대기열 측정 불가 |
| `avg_charge_wait_nan` | 충전 대기 시간 측정 불가 |
| `battery_std_nan` | 배터리 편차 측정 불가 |
| `robot_utilization_nan` | 가동률 측정 불가 |
| `blocked_path_15m_nan` | 경로 차단 집계 누락 |

---

### 8. 혼잡·밀도 상호작용

혼잡도와 다른 위험 지표의 곱으로 복합 위험 강도 표현.

| 피처 | 수식 | 의도 |
|---|---|---|
| `congestion_x_density` | congestion_score × max_zone_density | 혼잡 + 밀집 동시 발생 |
| `collision_x_density` | near_collision_15m × max_zone_density | 충돌 위험 × 밀집도 |
| `blocked_x_congestion` | blocked_path_15m × congestion_score | 경로 차단 × 혼잡 |
| `blocked_x_collision` | blocked_path_15m × near_collision_15m | 경로 차단 × 충돌 위험 |
| `fault_x_congestion` | fault_count_15m × congestion_score | 장애 × 혼잡 |
| `fault_x_density` | fault_count_15m × max_zone_density | 장애 × 밀집도 |
| `intersection_x_congestion` | intersection_count × congestion_score | 교차로 수 × 혼잡 |
| `congestion_per_intersection` | congestion_score / intersection_count | 교차로당 혼잡도 |

---

### 9. 주문 처리량 피처

수요 압력을 가용 자원 대비로 표현.

| 피처 | 수식 | 의도 |
|---|---|---|
| `order_per_pack_station` | order_inflow_15m / pack_station_count | 포장 스테이션당 주문 부하 |
| `order_per_robot` | order_inflow_15m / robot_total | 로봇 1대당 주문 |
| `order_per_active_robot` | order_inflow_15m / robot_active | 실제 작업 로봇 1대당 주문 |
| `sku_per_order` | unique_sku_15m / order_inflow_15m | 주문당 SKU 다양성 |
| `sku_concentration` | order_inflow_15m / unique_sku_15m | SKU 집중도 |
| `pack_utilization` | order_inflow_15m / pack_station_count | 포장 설비 활용률 |
| `pack_util_x_order` | pack_utilization × order_inflow_15m | 포장 부하 × 주문량 |
| `pack_util_x_congestion` | pack_utilization × congestion_score | 포장 부하 × 혼잡 |

---

### 10. 배터리·충전 피처

충전 부하와 배터리 위험의 복합 압력 표현.

| 피처 | 수식 | 의도 |
|---|---|---|
| `battery_risk_score` | low_battery_ratio × robot_idle | 배터리 부족 + 대기 로봇 |
| `charging_pressure` | low_battery_ratio × robot_charging | 배터리 위험 × 충전 중 로봇 |
| `battery_x_congestion` | low_battery_ratio × congestion_score | 배터리 위험 × 혼잡 |
| `charging_robot_per_charger` | robot_charging / charger_count | 충전기 1대당 충전 로봇 수 |
| `queue_pressure` | charge_queue_length × charging_ratio | 충전 대기열 × 충전 비율 |
| `charging_x_pack_util` | charging_ratio × pack_utilization | 충전 부하 × 포장 부하 |
| `charging_x_congestion` | charging_ratio × congestion_score | 충전 부하 × 혼잡 |
| `charging_x_order_pressure` | charging_ratio × order_per_active_robot | 충전 부하 × 실질 주문 압력 |
| `charging_x_blocked` | charging_ratio × blocked_path_15m | 충전 부하 × 경로 차단 |
| `charging_x_battery` | charging_ratio × low_battery_ratio | 충전 부하 × 배터리 위험 |

---

### 11. 레이아웃·이동 피처

| 피처 | 수식 | 의도 |
|---|---|---|
| `trip_x_recovery` | avg_trip_distance × avg_recovery_time | 이동 거리 × 회복 시간 |
| `layout_x_density` | layout_compactness × max_zone_density | 레이아웃 밀집도 × 구역 밀집도 |
| `dispersion_x_density` | zone_dispersion × max_zone_density | 구역 분산 × 밀집도 |
| `charging_x_layout_density` | charging_ratio × layout_x_density | 충전 부하 × 공간 밀집도 |

---

### 12. availability_ratio 상호작용 (v17)

| 피처 | 수식 | 의도 |
|---|---|---|
| `avail_x_congestion` | availability_ratio × congestion_score | 가용 로봇이 많아도 혼잡한 상황 |
| `avail_x_order_pressure` | availability_ratio × order_per_active_robot | 가용 로봇 대비 주문 압력 |
| `avail_x_low_battery` | availability_ratio × low_battery_ratio | 가용 로봇 중 배터리 위험 |

---

### 13. avg_trip_distance 상호작용 (v17)

| 피처 | 수식 | 의도 |
|---|---|---|
| `trip_x_congestion` | avg_trip_distance × congestion_score | 이동 거리 × 혼잡 |
| `trip_x_density` | avg_trip_distance × max_zone_density | 이동 거리 × 밀집도 |
| `trip_x_order` | avg_trip_distance × order_per_active_robot | 이동 거리 × 주문 압력 |

---

### 14. 미사용 컬럼 상호작용 (v17)

원본 데이터에 있으나 단독으로 활용이 적던 컬럼들의 interaction.

| 피처 | 수식 |
|---|---|
| `path_opt_x_congestion` | path_optimization_score × congestion_score |
| `intersection_wait_x_count` | intersection_wait_time_avg × intersection_count |
| `charge_eff_x_charging` | charge_efficiency_pct × charging_ratio |
| `wms_x_order` | wms_response_time_ms × order_inflow_15m |
| `agv_success_x_util` | agv_task_success_rate × robot_utilization |

---

### 15. 로봇 수 차이·합

| 피처 | 수식 | 의도 |
|---|---|---|
| `idle_minus_active` | robot_idle - robot_active | 대기 > 작업이면 양수 (여유 과잉) |
| `idle_plus_charging` | robot_idle + robot_charging | 작업 외 로봇 수 |

---

### 16. 복합 지연 위험 지수

여러 위험 요소를 하나로 결합한 종합 압력 지표.

| 피처 | 수식 |
|---|---|
| `delay_risk_proxy` | order_per_active_robot × (1 + congestion_score) × (1 + low_battery_ratio) |
| `charging_x_delay_risk` | charging_ratio × delay_risk_proxy |

---

### 17. 교대 시간 사이클 인코딩

`shift_hour`는 24시간 주기이므로 sin/cos로 변환해 순환성 표현.

| 피처 | 수식 |
|---|---|
| `shift_hour_sin` | sin(2π × shift_hour / 24) |
| `shift_hour_cos` | cos(2π × shift_hour / 24) |
| `charging_x_shift_sin` | charging_ratio × shift_hour_sin |

---

### 18. Scene × 현재 상태 상호작용

"현재 타임슬롯이 이 시나리오 평균 대비 얼마나 나쁜가"를 수치화.

| 피처 | 수식 | 의도 |
|---|---|---|
| `charging_vs_scene_mean` | charging_ratio - scene_charging_ratio_raw_mean | 현재 충전 부하가 시나리오 평균보다 높은 정도 |
| `charging_vs_scene_ratio` | charging_ratio / scene_charging_ratio_raw_mean | 시나리오 평균 대비 배율 |
| `congestion_vs_scene_mean` | congestion_score - scene_congestion_score_mean | 현재 혼잡도가 시나리오 평균보다 높은 정도 |
| `order_vs_scene_mean` | order_inflow_15m - scene_order_inflow_15m_mean | 현재 주문량이 시나리오 평균보다 높은 정도 |
| `order_vs_scene_ratio` | order_inflow_15m / scene_order_inflow_15m_mean | 시나리오 평균 대비 배율 |
| `utilization_vs_scene_mean` | robot_utilization - scene_robot_utilization_mean | 현재 가동률이 시나리오 평균보다 높은 정도 |
| `battery_vs_scene_mean` | low_battery_ratio - scene_low_battery_ratio_mean | 현재 배터리 위험이 시나리오 평균보다 높은 정도 |
| `trip_vs_scene_mean` | avg_trip_distance - scene_avg_trip_distance_mean | 현재 이동 거리가 시나리오 평균보다 높은 정도 |
| `trip_vs_scene_ratio` | avg_trip_distance / scene_avg_trip_distance_mean | 시나리오 평균 대비 배율 |

---

### 19. OOF Target Encoding

범주형 컬럼을 "그 카테고리의 평균 지연시간"으로 치환. leakage 방지를 위해 OOF 방식으로 계산.

| 피처 | 기준 | 의도 |
|---|---|---|
| `te__layout_id` | layout_id | 해당 레이아웃의 역사적 평균 지연 |
| `te__scenario_id` | scenario_id | 해당 시나리오의 역사적 평균 지연 |
| `te__layout_type` | layout_type | 레이아웃 타입별 평균 지연 |
| `te__layout_cluster` | layout_cluster | 클러스터별 평균 지연 (unseen layout 대응) |
| `te__layout_scenario` | layout_id + scenario_id | 레이아웃-시나리오 조합의 평균 지연 |
| `te__cluster_type` | layout_cluster + layout_type | 클러스터-타입 조합의 평균 지연 |

---

## 시도했으나 최종 제거된 피처

| 피처 그룹 | 도입 | 제거 이유 |
|---|---|---|
| **SCENE_COLS 6개 추가** (v18) | v18 | CV↑ Dacon↓, 과적합 피처로 확정 |
| **lag / diff 피처** (v18) | v18 | 순서 의존적 + Dacon 악화. 정렬 버그 발생 |
| **Bayesian TE** (v18) | v18 | simple mean 대비 정보 손실, Dacon 악화 |
| **scene × avail_ratio 교차** (v18) | v18 | Dacon 개선 없음, v23에서 제거 |
| **scenario percentile rank** (v19) | v19 | CV↑ Dacon↓, CV 전용 신호 |
| **MLP** (v19) | v19 | CV 10.13으로 트리 대비 현저히 낮음 |
| **layout context** (v25) | v25 | layout_mean 계산에 val fold 값 포함 → CV leakage |
| **lead features** (v26) | v26 | 피처셋 과부하로 Dacon 악화 (신호 자체는 유효) |
| **svl features** (v26) | v26 | 동일 이유, CV leakage 내재 |
