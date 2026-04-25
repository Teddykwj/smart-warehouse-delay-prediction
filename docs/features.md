# 파생 피처 설명

원본 데이터에 없는, 코드에서 새로 만든 피처들 정리.

---

## 1. Layout Clustering

레이아웃의 물리적 특성으로 KMeans 클러스터링.

| 피처 | 수식 | 의도 |
|---|---|---|
| `layout_cluster` | KMeans(레이아웃 수치 컬럼들) | 비슷한 구조의 레이아웃을 묶어 unseen layout도 TE 혜택 받게 함 |

---

## 2. Scenario Context Features

각 scenario의 25개 timeslot 전체를 집계해 "이 시나리오 전반의 상황"을 요약.  
컬럼당 5개 통계 생성 (mean / std / max / min / range).

| 피처 패턴 | 집계 대상 | 의도 |
|---|---|---|
| `scene_charging_ratio_raw_{stat}` | 시나리오 내 충전 비율 분포 | 이 시나리오가 전반적으로 충전 부하가 높았는지 |
| `scene_congestion_score_{stat}` | 시나리오 내 혼잡도 분포 | 전반적인 혼잡 수준 |
| `scene_low_battery_ratio_{stat}` | 시나리오 내 배터리 위험 비율 분포 | 배터리 상황이 얼마나 나빴는지 |
| `scene_order_inflow_15m_{stat}` | 시나리오 내 주문량 분포 | 전반적인 수요 압력 |
| `scene_robot_utilization_{stat}` | 시나리오 내 가동률 분포 | 로봇 부하 수준 |

`{stat}` = mean / std / max / min / range

---

## 3. Robot State Ratios

로봇 상태 원시값을 전체 로봇 수 대비 비율로 변환.

| 피처 | 수식 | 의도 |
|---|---|---|
| `idle_ratio` | robot_idle / robot_total | 대기 중 로봇 비율 |
| `charging_ratio` | robot_charging / robot_total | 충전 중 로봇 비율 (핵심 피처) |
| `active_ratio` | robot_active / robot_total | 작업 중 로봇 비율 |
| `utilization_gap` | robot_idle / robot_active | 대기 대비 작업 비율 (괴리도) |
| `robot_available` | robot_active + robot_idle | 즉시 투입 가능한 로봇 수 |
| `availability_ratio` | robot_available / robot_total | 가용 로봇 비율 |

---

## 4. charging_ratio 다항식

`charging_ratio`가 지연에 비선형적 영향을 주기 때문에 다항식으로 확장.

| 피처 | 수식 |
|---|---|
| `charging_ratio_sq` | charging_ratio² |
| `charging_ratio_cube` | charging_ratio³ |
| `charging_ratio_log1p` | log(1 + charging_ratio) |

---

## 5. availability_ratio 다항식 (v16)

`availability_ratio`가 피처 중요도 30.7%로 상위권이었으나 다항식이 없었던 것을 추가.

| 피처 | 수식 |
|---|---|
| `availability_ratio_sq` | availability_ratio² |
| `availability_ratio_log1p` | log(1 + availability_ratio) |
| `availability_ratio_inv` | 1 / availability_ratio |

---

## 6. NaN 지시자

결측 자체가 운영 이상 신호일 수 있어 별도 피처로 추가.

| 피처 | 의미 |
|---|---|
| `congestion_score_nan` | 혼잡도 측정 불가 여부 |
| `fault_count_15m_nan` | 장애 집계 누락 여부 |
| `near_collision_15m_nan` | 충돌 위험 집계 누락 여부 |
| `avg_recovery_time_nan` | 회복 시간 측정 불가 여부 |
| `charge_queue_length_nan` | 충전 대기열 측정 불가 여부 |
| `avg_charge_wait_nan` | 충전 대기 시간 측정 불가 여부 |
| `battery_std_nan` | 배터리 편차 측정 불가 여부 |
| `robot_utilization_nan` | 가동률 측정 불가 여부 |
| `blocked_path_15m_nan` | 경로 차단 집계 누락 여부 |

---

## 7. 혼잡·밀도 상호작용

혼잡도와 다른 위험 지표의 곱으로 복합 위험 강도를 표현.

| 피처 | 수식 | 의도 |
|---|---|---|
| `congestion_x_density` | congestion_score × max_zone_density | 혼잡 + 밀집 구역 동시 발생 |
| `collision_x_density` | near_collision_15m × max_zone_density | 충돌 위험 × 밀집도 |
| `blocked_x_congestion` | blocked_path_15m × congestion_score | 경로 차단 × 혼잡 |
| `blocked_x_collision` | blocked_path_15m × near_collision_15m | 경로 차단 × 충돌 위험 |
| `fault_x_congestion` | fault_count_15m × congestion_score | 장애 × 혼잡 |
| `fault_x_density` | fault_count_15m × max_zone_density | 장애 × 밀집도 |
| `intersection_x_congestion` | intersection_count × congestion_score | 교차로 수 × 혼잡 |
| `congestion_per_intersection` | congestion_score / intersection_count | 교차로당 혼잡도 |

---

## 8. 주문 처리량 피처

수요 압력을 가용 자원 대비로 표현.

| 피처 | 수식 | 의도 |
|---|---|---|
| `order_per_pack_station` | order_inflow_15m / pack_station_count | 포장 스테이션당 주문 부하 |
| `order_per_robot` | order_inflow_15m / robot_total | 로봇 1대당 주문 |
| `order_per_active_robot` | order_inflow_15m / robot_active | 실제 작업 로봇 1대당 주문 (실질 부하) |
| `sku_per_order` | unique_sku_15m / order_inflow_15m | 주문당 SKU 다양성 |
| `sku_concentration` | order_inflow_15m / unique_sku_15m | SKU 집중도 (역수 관계) |
| `pack_utilization` | order_inflow_15m / pack_station_count | 포장 설비 활용률 |
| `pack_util_x_order` | pack_utilization × order_inflow_15m | 포장 부하 × 주문량 |
| `pack_util_x_congestion` | pack_utilization × congestion_score | 포장 부하 × 혼잡 |

---

## 9. 배터리·충전 피처

충전 부하와 배터리 위험의 복합 압력 표현.

| 피처 | 수식 | 의도 |
|---|---|---|
| `battery_risk_score` | low_battery_ratio × robot_idle | 배터리 부족 + 대기 로봇 (충전 대기 압력) |
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

## 10. 레이아웃·이동 피처

공간 구조와 이동 효율의 결합.

| 피처 | 수식 | 의도 |
|---|---|---|
| `trip_x_recovery` | avg_trip_distance × avg_recovery_time | 이동 거리 × 회복 시간 (이동 비효율) |
| `layout_x_density` | layout_compactness × max_zone_density | 레이아웃 밀집도 × 구역 밀집도 |
| `dispersion_x_density` | zone_dispersion × max_zone_density | 구역 분산 × 밀집도 |
| `charging_x_layout_density` | charging_ratio × layout_x_density | 충전 부하 × 공간 밀집도 |

---

## 11. 로봇 수 차이·합

| 피처 | 수식 | 의도 |
|---|---|---|
| `idle_minus_active` | robot_idle - robot_active | 대기 > 작업이면 양수 (여유 과잉) |
| `idle_plus_charging` | robot_idle + robot_charging | 작업 외 로봇 수 |

---

## 12. 복합 지연 위험 지수

여러 위험 요소를 하나로 결합한 종합 압력 지표.

| 피처 | 수식 |
|---|---|
| `delay_risk_proxy` | order_per_active_robot × (1 + congestion_score) × (1 + low_battery_ratio) |
| `charging_x_delay_risk` | charging_ratio × delay_risk_proxy |

주문 압력 × 혼잡 보정 × 배터리 위험 보정을 곱해 지연 가능성을 하나의 숫자로 표현.

---

## 13. 교대 시간 사이클 인코딩

`shift_hour`는 24시간 주기를 가지므로 숫자 그대로 쓰면 23→0 점프가 발생. sin/cos로 변환.

| 피처 | 수식 |
|---|---|
| `shift_hour_sin` | sin(2π × shift_hour / 24) |
| `shift_hour_cos` | cos(2π × shift_hour / 24) |
| `charging_x_shift_sin` | charging_ratio × shift_hour_sin |

---

## 14. Scenario Context × 현재 상태 상호작용 (v16)

"현재 timeslot이 이 시나리오 평균 대비 얼마나 나쁜가?"를 수치화.

| 피처 | 수식 | 의도 |
|---|---|---|
| `charging_vs_scene_mean` | charging_ratio - scene_charging_ratio_raw_mean | 현재 충전 부하가 시나리오 평균보다 높은 정도 |
| `charging_vs_scene_ratio` | charging_ratio / scene_charging_ratio_raw_mean | 현재 충전 부하의 시나리오 평균 대비 배율 |
| `congestion_vs_scene_mean` | congestion_score - scene_congestion_score_mean | 현재 혼잡도가 시나리오 평균보다 높은 정도 |

---

## 15. OOF Target Encoding 피처

범주형 컬럼을 "그 카테고리의 평균 지연시간"으로 치환. leakage 방지를 위해 OOF 방식으로 계산.

| 피처 | 기준 컬럼 | 의도 |
|---|---|---|
| `te__layout_id` | layout_id | 해당 레이아웃의 역사적 평균 지연 |
| `te__scenario_id` | scenario_id | 해당 시나리오의 역사적 평균 지연 |
| `te__layout_type` | layout_type | 레이아웃 타입별 평균 지연 |
| `te__layout_cluster` | layout_cluster | 클러스터별 평균 지연 (unseen layout 대응) |
| `te__layout_scenario` | layout_id + scenario_id 조합 | 레이아웃-시나리오 조합의 평균 지연 |
| `te__cluster_type` | layout_cluster + layout_type 조합 | 클러스터-타입 조합의 평균 지연 (unseen layout 대응) |
