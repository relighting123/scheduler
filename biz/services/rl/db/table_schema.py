"""스케줄러 RL 입력/출력 테이블 DDL."""

from biz.services.rl.db.linedb_transform import LINEDB_TABLE


def create_learning_tables(db):
    """RTS_LINEDSDB_INF 단일 EAV 입력 테이블 생성."""
    db.execute(f"""
        CREATE TABLE {LINEDB_TABLE} (
            RULE_TIMEKEY VARCHAR2(50) NOT NULL,
            FAC_ID VARCHAR2(50) NOT NULL,
            BATCH_ID VARCHAR2(50) NOT NULL,
            PLAN_PROD_KEY VARCHAR2(200) NOT NULL,
            OPER_ID VARCHAR2(50) NOT NULL,
            OPER_SEQ NUMBER,
            EQP_MODEL_CD VARCHAR2(50) NOT NULL,
            GBN_CD VARCHAR2(50) NOT NULL,
            ATTR_VAL VARCHAR2(50),
            CONSTRAINT PK_RTS_LINEDSDB_INF PRIMARY KEY (
                RULE_TIMEKEY, FAC_ID, BATCH_ID, PLAN_PROD_KEY,
                OPER_ID, EQP_MODEL_CD, GBN_CD
            )
        )
    """)


def create_output_tables(db):
    """추론 결과 출력 테이블 생성."""
    db.execute("""
        CREATE TABLE RTD_CONV_INF (
            RULE_TIMEKEY VARCHAR2(20),
            FROM_BATCH VARCHAR2(50),
            FROM_PLAN_PROD_KEY VARCHAR2(50),
            FROM_OPER_ID VARCHAR2(50),
            EQP_MODEL_CD VARCHAR2(50),
            TO_BATCH_ID VARCHAR2(50),
            TO_PLAN_PROD_KEY VARCHAR2(50),
            TO_OPER_ID VARCHAR2(50),
            START_CONV_TIME VARCHAR2(20),
            EQP_QTY NUMBER
        )
    """)
    db.execute("""
        CREATE TABLE RTS_RSLT_MAS (
            RULE_TIMEKEY VARCHAR2(50) NOT NULL,
            SEQ_NO NUMBER NOT NULL,
            EQP_ID VARCHAR2(50) NOT NULL,
            EQP_MODEL_CD VARCHAR2(50),
            BATCH_ID VARCHAR2(50),
            START_TM VARCHAR2(14),
            END_TM VARCHAR2(14),
            PLAN_PROD_ATTR_VAL VARCHAR2(200),
            PROD_QTY VARCHAR2(50),
            CUM_PROD_QTY VARCHAR2(50),
            CRT_USET_ID VARCHAR2(50),
            CRT_TM VARCHAR2(14),
            CONSTRAINT PK_RTS_RSLT_MAS PRIMARY KEY (RULE_TIMEKEY, EQP_ID, SEQ_NO)
        )
    """)
