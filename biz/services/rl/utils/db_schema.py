"""DDL for scheduler RL input/output tables."""


def create_learning_tables(db):
    """Create the seven RULE_TIMEKEY-scoped input tables."""
    db.execute("""
        CREATE TABLE WIP_INFO (
            RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
            OPER_ID VARCHAR2(50), OPER_SEQ NUMBER, WIP_QTY NUMBER
        )
    """)
    db.execute("""
        CREATE TABLE UPH_INFO (
            RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
            OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), UPH NUMBER
        )
    """)
    db.execute("""
        CREATE TABLE EQP_QTY_INFO (
            RULE_TIMEKEY VARCHAR2(14), BATCH_ID VARCHAR2(50),
            EQP_MODEL_CD VARCHAR2(50), TIME_SLOT VARCHAR2(50), EQP_QTY NUMBER
        )
    """)
    db.execute("""
        CREATE TABLE AVAIL_INFO (
            RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
            OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), AVAIL_YN VARCHAR2(10)
        )
    """)
    db.execute("""
        CREATE TABLE BATCH_TOOL_INFO (
            RULE_TIMEKEY VARCHAR2(14), BATCH_ID VARCHAR2(50),
            PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50)
        )
    """)
    db.execute("""
        CREATE TABLE TOOL_QTY_INFO (
            RULE_TIMEKEY VARCHAR2(14), BATCH_ID VARCHAR2(50),
            EQP_MODEL_CD VARCHAR2(50), TOOL_QTY NUMBER
        )
    """)
    db.execute("""
        CREATE TABLE PLAN_INFO (
            RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
            OPER_ID VARCHAR2(50), START_TIME VARCHAR2(50),
            END_TIME VARCHAR2(50), PLAN_QTY NUMBER
        )
    """)


def create_benchmark_tables(db):
    """Benchmark scenario registry (input rows live in the seven input tables)."""
    db.execute("""
        CREATE TABLE BENCHMARK_SCENARIO (
            SCENARIO_ID VARCHAR2(100) NOT NULL,
            RULE_TIMEKEY VARCHAR2(14) NOT NULL,
            DESCRIPTION VARCHAR2(500),
            GROUND_TRUTH_JSON CLOB,
            CONSTRAINT PK_BENCHMARK_SCENARIO PRIMARY KEY (SCENARIO_ID)
        )
    """)


def create_output_tables(db):
    """Create inference output tables."""
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
