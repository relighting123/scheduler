# 계획 기반 장비 배치 분석

Input 스냅샷 7종만 사용해, 시뮬레이터·강화학습 없이 **계획제품×공정×장비모델** 대수가 전체 계획 달성에 얼마나 기여하는지 정적으로 분석·재배치합니다.

## Input 7종

| 데이터 | 설명 |
|--------|------|
| wip_info | 재공, 공정 순서 |
| uph_info | 모델별 시간당 생산량 |
| eqp_qty_info | 배치·모델별 장비 대수(풀) |
| avail_info | 처리 가능 여부 |
| batch_tool_info | 배치–제품–공정 매핑 |
| tool_qty_info | Tool 상한 |
| plan_info | 계획량·구간 |

DB는 EAV 테이블 `RTS_LINEDSDB_INF`에서 조회할 수 있습니다.

## 실행

```bash
# CSV 벤치마크 (DB 불필요)
python3 run.py --benchmark-dataset benchmark_dataset

# 현재 배치만 분석
python3 run.py --benchmark-dataset benchmark_dataset --no-optimize

# API
# POST /tasks  { "action": "plan_allocation", "parameters": { "scenario": "benchmark_dataset" } }
```

## 프로젝트 구조

- `core/` — FastAPI, Queue, DB (Oracle)
- `biz/services/plan_allocation/` — 정적 용량·조합 최적화·리포트
- `biz/services/plan_allocation_service.py` — 진입점
- `test/data/benchmark_dataset/` — 샘플 Input CSV

## 비즈니스 로직 추가

1. `biz/services/`에 서비스 모듈 추가
2. `biz/task_processor.py`에 `action` 분기 등록
3. `/tasks`로 호출

예제: `examples/add_task_example.py`
