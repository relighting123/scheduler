# 조합최적화 벤치마크 테스트 데이터

학습 후 성능 비교에 사용하는 테스트 세트입니다. DB 없이도 `TestDataLoader`로 로드할 수 있습니다.

## 디렉터리 구조

| 파일 | 설명 |
|------|------|
| `wip_info.csv` | 재공(WIP) 스냅샷 |
| `uph_info.csv` | 장비 모델별 UPH |
| `eqp_qty_info.csv` | 배치·시간대별 장비 대수 |
| `avail_info.csv` | 처리 가능 여부 |
| `batch_tool_info.csv` | 배치–제품/공정 매핑 |
| `tool_qty_info.csv` | Tool 수량 |
| `plan_info.csv` | 계획 수량 |
| `ground_truth.json` | 정답(Optimal) 기준 기대 지표 |

모든 입력 CSV는 `RULE_TIMEKEY` 컬럼을 포함합니다 (형식: `YYYYMMDDHHMMSS`).

## ground_truth.json

- `optimal`: OptimalExpert 기준 OP20 계획달성률 및 장비 전환 횟수 (벤치마크 정답)
- `evaluation`: 테스트 시 RL·휴리스틱 결과를 위 값과 비교
