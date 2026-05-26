# 벤치마크 데이터셋 (Benchmark Dataset)

학습 후 성능 검증·오프라인 평가에 사용하는 표준 테스트 데이터셋입니다.

## 구성

| 파일 | 설명 |
|------|------|
| `wip_info.csv` ~ `plan_info.csv` | Input 스냅샷 7종 (`RULE_TIMEKEY` 포함) |
| `ground_truth.json` | 기대 지표(Optimal 기준) 및 목표 장비 배치 |

## 로드

```python
from core.repository import BaseRepository
from biz.services.rl_scheduler_service import RLSchedulerService

service = RLSchedulerService(db_manager=BaseRepository())
service.seed_benchmark_scenarios(scenarios=["benchmark_dataset"])
data = service.data.fetch_data(rule_timekey="20251020070000")
```

시나리오 ID: `benchmark_dataset`
