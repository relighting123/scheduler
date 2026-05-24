# 벤치마크 시나리오 (`test/data`)

## ID 규칙

`bench_{순번2자리}_{slug}` — 예: `bench_01_multiproduct`

레거시 ID는 `biz.services.rl.benchmark_scenarios.resolve_scenario_id()`로 자동 매핑됩니다.

| 시나리오 ID | 설명 | 폴더 |
|-------------|------|------|
| `bench_01_multiproduct` | 다품종·다모델 표준 벤치마크 | `bench_01_multiproduct/` |
| `bench_02_initial_conv` | 초기 장비 전환(재배치) 필수 | `bench_02_initial_conv/` |

### 레거시 별칭 (하위 호환)

| 구 ID | 신 ID |
|-------|--------|
| `benchmark_dataset` | `bench_01_multiproduct` |
| `benchmark_initial_conv` | `bench_02_initial_conv` |

## 폴더 구성

각 시나리오 디렉터리:

- 입력 CSV 7종 (`wip_info.csv` … `plan_info.csv`, `RULE_TIMEKEY` 포함)
- `ground_truth.json` — `scenario_id`, `rule_timekey`, `target_allocation`, 평가 지표

## 로드

```python
from biz.services.rl.test_data_loader import TestDataLoader

loader = TestDataLoader()
print(loader.list_scenarios())
data = loader.load_for_env("bench_01_multiproduct")
```

## CLI

```bash
python run.py train --benchmark-dataset bench_01_multiproduct
python run.py train --benchmark-dataset bench_02_initial_conv
python test_bench_01_multiproduct.py
python test_bench_02_initial_conv.py
```
