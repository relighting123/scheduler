# 벤치마크: 초기 장비 전환 필수 (`benchmark_initial_conv`)

조합 최적화형 난이도 검증용 시나리오입니다. **초기 배치가 의도적으로 잘못**되어 있어, 초기 구간에 장비 재배치를 하지 않으면 계획 달성이 크게 떨어집니다. 정답 배치로 맞춘 뒤 유지하면 고달성이 나와야 “잘 푼 것”과 “그냥 둔 것”을 구분할 수 있습니다.

## 설계 요약

| 항목 | 내용 |
|------|------|
| 초기(잘못된) 배치 | MODEL_A 8대 → P1\|OP10 집중, MODEL_B 8대 → P3\|OP10, MODEL_C 5대 → P3\|OP20 |
| 병목 | **P2** 공정에 MODEL_B 가용·UPH는 높으나 초기 배치 0대 |
| 정답 배치 | `ground_truth.json` → `target_allocation` |
| 난이도 | 모델별 **AVAIL_YN**·**UPH** 상이 (P1=A, P2=B 전용, P3=C/B) |

## 기대 성능 (24 step, 시뮬레이션 검증 기준)

| 정책 | P1 OP20 | P2 OP20 | P3 OP20 | 평균 | 전환 |
|------|---------|---------|---------|------|------|
| 전환 없음 | ~0% | ~0% | ~120% | ~40% | 0 |
| Optimal 초기 재배치 후 유지 | ~98% | ~96% | ~120% | ~105% | ~20 |

## 로드

```python
from core.repository import BaseRepository
from biz.services.rl_scheduler_service import RLSchedulerService

service = RLSchedulerService(db_manager=BaseRepository())
service.seed_benchmark_scenarios(scenarios=["benchmark_initial_conv"])
data = service.data.fetch_data(rule_timekey="20251020100000")
```

## CLI

```bash
python run.py train --benchmark-dataset benchmark_initial_conv
python test_benchmark_initial_conv.py
```
