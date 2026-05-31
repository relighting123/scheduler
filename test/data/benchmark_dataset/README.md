# 벤치마크 데이터셋

계획 배치 분석·단위 테스트용 Input CSV 샘플입니다.

## 구성

| 파일 | 설명 |
|------|------|
| `wip_info.csv` ~ `plan_info.csv` | Input 스냅샷 7종 |
| `ground_truth.json` | 참고용 목표 배치·지표 |

## 로드

```python
from biz.services.plan_allocation.test_data_loader import TestDataLoader
data = TestDataLoader().load_snapshot("benchmark_dataset")
```
