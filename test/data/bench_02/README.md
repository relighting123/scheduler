# 벤치마크 `bench_02` — 초기 장비 전환 필수

조합 최적화형 난이도 검증용 시나리오입니다. **초기 배치가 의도적으로 잘못**되어 있어, 초기 구간에 장비 재배치를 하지 않으면 계획 달성이 크게 떨어집니다.

## 설계 요약

| 항목 | 내용 |
|------|------|
| 초기(잘못된) 배치 | MODEL_A 8대 → P1\|OP10 집중, MODEL_B 8대 → P3\|OP10, MODEL_C 5대 → P3\|OP20 |
| 병목 | **P2** 공정에 MODEL_B 가용·UPH는 높으나 초기 배치 0대 |
| 정답 배치 | `ground_truth.json` → `target_allocation` |

## 로드

```python
from biz.services.rl.test_data_loader import TestDataLoader
data = TestDataLoader().load_for_env("bench_02")
```

## CLI

```bash
python run.py benchmark --benchmark-dataset bench_02
python test_bench_02.py
```

시나리오 ID: `bench_02`
