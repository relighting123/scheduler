# 벤치마크: 다품종·다모델 표준 (`bench_01_multiproduct`)

학습 후 성능 검증·오프라인 평가에 사용하는 표준 테스트 데이터셋입니다.

## 구성

| 파일 | 설명 |
|------|------|
| `wip_info.csv` ~ `plan_info.csv` | Input 스냅샷 7종 (`RULE_TIMEKEY` 포함) |
| `ground_truth.json` | 기대 지표(Optimal 기준) 및 목표 장비 배치 |

## 로드

```python
from biz.services.rl.test_data_loader import TestDataLoader
data = TestDataLoader().load_for_env("bench_01_multiproduct")
```

시나리오 ID: `bench_01_multiproduct` (구 `benchmark_dataset`)
