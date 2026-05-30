# `biz/services/rl/db` — DB 레이어

스케줄러 RL의 **Oracle 테이블 DDL**, **RTS_LINEDSDB_INF(EAV) 조회·변환**, **학습용 스냅샷 로드**를 담당한다.  
환경(`SchedulerEnv`) 생성·추론 헬퍼는 DB와 무관하므로 `biz/services/rl/env/env_factory.py`에 둔다.

## 폴더 구조

| 파일 | 역할 |
|------|------|
| `ddl.py` | `CREATE TABLE` DDL — 입력 `RTS_LINEDSDB_INF`, 출력 `RTD_CONV_INF` / `RTS_RSLT_MAS` |
| `linedb_constants.py` | EAV 테이블명·컬럼·`GBN_CD` 상수, D0/D1 계획 구간, 빈 스냅샷 DataFrame |
| `linedb_snapshot_sql.py` | EAV → env 7종 스냅샷 **SQL 조회** (기본 경로) |
| `linedb_snapshot_pandas.py` | EAV DataFrame → env 7종 **pandas 변환** (DB 실패 시 폴백) |
| `training_data_access.py` | `TrainingDataAccess` — RULE_TIMEKEY 해석·범위 조회·스냅샷 로드 |

## 데이터 흐름

```
RTS_LINEDSDB_INF (EAV)
        │
        ├─► linedb_snapshot_sql.fetch_snapshot_from_db()  ──► wip/uph/plan 등 7 DataFrame
        │
        └─► (폴백) training_data_access → linedb_snapshot_pandas.transform_linedb_snapshot()
```

## RTS_LINEDSDB_INF 스키마

테이블 명: `RTS_LINEDSDB_INF`

| 컬럼 | 타입 | PK | 설명 |
|------|------|-----|------|
| RULE_TIMEKEY | VARCHAR2(50) | O | 스냅샷 시각 키 |
| FAC_ID | VARCHAR2(50) | O | 공장 |
| BATCH_ID | VARCHAR2(50) | O | 배치 |
| PLAN_PROD_KEY | VARCHAR2(200) | O | 계획 제품 키 |
| OPER_ID | VARCHAR2(50) | O | 공정 |
| OPER_SEQ | NUMBER | | 공정 순서 |
| EQP_MODEL_CD | VARCHAR2(50) | O | 설비 모델 |
| GBN_CD | VARCHAR2(50) | O | 지표 구분 |
| ATTR_VAL | VARCHAR2(50) | | 값 |

### GBN_CD ↔ env 스냅샷

| GBN_CD | env 출력 |
|--------|----------|
| WIP_QTY | `wip_info` |
| UPH | `uph_info`, `avail_info` (UPH 없으면 AVAIL_YN=N, 진행 불가) |
| ASSIGN_EQUIP_CNT | `eqp_qty_info` |
| TOOL_QTY | `tool_qty_info` |
| D0_TARGET_QTY / D1_TARGET_QTY | `plan_info` (07시 경계, `plan_windows_for_rule_timekey`) |
| (차원 조합) | `batch_tool_info` |

### 데이터 예시

- RULE_TIMEKEY: `"2026052922500000"`
- FAC_ID: `["ICPRB","CJPRB"]`
- BATCH_ID: `["9C/92","9C/102", ...]`
- PLAN_PROD_KEY: `["M15/59C/H5UDGSTED/E1S/NA"]`
- OPER_ID: `["Z1020000A","Z1040000A"]`
- OPER_SEQ: 1, 2, 3, ...
- EQP_MODEL_CD: `["T5833","MAGNUM5"]`
- GBN_CD: `ASSIGN_EQUIP_CNT`, `UPH`, `WIP_QTY`, `D0_TARGET_QTY`, `D1_TARGET_QTY`, `TOOL_QTY`
- ATTR_VAL: 해당 GBN_CD의 수치

**PLAN_PROD_KEY/EQP_MODEL 기준 조회 시 UPH가 없으면 진행 불가로 판단한다.**

- **D0_TARGET**: RULE_TIMEKEY 시점 ~ 다음날 07시까지 계획
- **D1_TARGET**: 다음날 07시 ~ 그 다음날 07시까지 계획
