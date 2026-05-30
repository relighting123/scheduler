"""Thin endpoint facade for scheduler RL operations."""

from biz.services.rl.db.ddl import create_learning_tables, create_output_tables
from biz.services.rl.db.training_data_access import DEFAULT_RULE_TIMEKEY, TrainingDataAccess
from biz.services.rl.env.env_factory import SchedulerEnvFactory
from biz.services.rl.infer.inference_runner import InferenceRunner
from biz.services.rl.train.trainer import BenchmarkTrainer
from biz.services.rl.validation.benchmark_evaluator import BenchmarkEvaluator


class RLSchedulerService:
    """Public API used by task processors and CLI entry points."""

    DEFAULT_RULE_TIMEKEY = DEFAULT_RULE_TIMEKEY

    def __init__(self, db_manager):
        self.db = db_manager
        self.data = TrainingDataAccess(db_manager, default_rule_timekey=self.DEFAULT_RULE_TIMEKEY)
        self.env_factory = SchedulerEnvFactory()
        self._trainer = BenchmarkTrainer(self)
        self._evaluator = BenchmarkEvaluator(self)
        self._infer = InferenceRunner(self)

    def fetch_data(self, rule_timekey=None):
        return self.data.fetch_data(rule_timekey=rule_timekey)

    def fetch_training_snapshots(
        self,
        from_rule_timekey=None,
        to_rule_timekey=None,
        rule_timekey=None,
    ):
        return self.data.fetch_training_snapshots(
            from_rule_timekey=from_rule_timekey,
            to_rule_timekey=to_rule_timekey,
            rule_timekey=rule_timekey,
        )

    def init_db_scenario(self):
        """Initialize heuristic trap scenario tables (DROP -> CREATE -> INSERT)."""
        from biz.services.rl.db.linedb_constants import (
            GBN_ASSIGN_EQUIP,
            GBN_D0_TARGET,
            GBN_TOOL,
            GBN_UPH,
            GBN_WIP,
            LINEDB_TABLE,
        )

        print("\n[DB 설정] DB 시나리오 초기화를 시작합니다 (Heuristic Trap Scenario)...")
        tables = [
            "WIP_INFO",
            "UPH_INFO",
            "EQP_QTY_INFO",
            "AVAIL_INFO",
            "BATCH_TOOL_INFO",
            "TOOL_QTY_INFO",
            "PLAN_INFO",
            LINEDB_TABLE,
            "RTD_CONV_INF",
            "RTS_RSLT_MAS",
        ]

        for table in tables:
            try:
                self.db.execute(f"DROP TABLE {table}")
            except Exception:
                pass

        create_learning_tables(self.db)
        create_output_tables(self.db)
        tk = self.DEFAULT_RULE_TIMEKEY
        fac = "FAC1"

        def insert_linedb(batch, prod, oper, oper_seq, model, gbn, val):
            self.db.execute(
                f"INSERT INTO {LINEDB_TABLE} VALUES ("
                f"'{tk}', '{fac}', '{batch}', '{prod}', '{oper}', "
                f"{oper_seq}, '{model}', '{gbn}', '{val}')"
            )

        insert_linedb("B1", "P1", "OP10", 10, "-", GBN_WIP, "5000")
        insert_linedb("B2", "P1", "OP20", 20, "-", GBN_WIP, "500")
        insert_linedb("B1", "P1", "OP10", 10, "MODEL_A", GBN_UPH, "100")
        insert_linedb("B2", "P1", "OP20", 20, "MODEL_A", GBN_UPH, "100")
        insert_linedb("B1", "P1", "OP10", 10, "MODEL_A", GBN_ASSIGN_EQUIP, "5")
        insert_linedb("B2", "P1", "OP20", 20, "MODEL_A", GBN_ASSIGN_EQUIP, "5")
        insert_linedb("B1", "P1", "OP10", 10, "MODEL_A", GBN_TOOL, "10")
        insert_linedb("B2", "P1", "OP20", 20, "MODEL_A", GBN_TOOL, "10")
        insert_linedb("B1", "P1", "OP10", 10, "-", GBN_D0_TARGET, "4000")
        insert_linedb("B2", "P1", "OP20", 20, "-", GBN_D0_TARGET, "4000")

        print(f"[성공] DB 시나리오 초기화가 완료되었습니다 ({LINEDB_TABLE}).")

    def generate_expert_data(self, env, num_samples=5000, expert_cls=None, target_allocation=None):
        return self._trainer.generate_expert_data(
            env,
            num_samples=num_samples,
            expert_cls=expert_cls,
            target_allocation=target_allocation,
        )

    def train_model(self, **kwargs):
        return self._trainer.train_model(**kwargs)

    def evaluate_on_benchmark_dataset(self, **kwargs):
        return self._evaluator.evaluate_dataset(**kwargs)

    def evaluate_all_benchmark_datasets(self, **kwargs):
        return self._evaluator.evaluate_all(**kwargs)

    def run_benchmark_evaluation(self, datasets=None, model_path="scheduler_ppo_model", max_steps=24):
        print("\n" + "=" * 80)
        print(" [벤치마크 데이터셋 검증 (Optimal vs Heuristic vs RL)]")
        print("=" * 80)
        return self._evaluator.evaluate_all(
            datasets=datasets,
            model_path=model_path,
            max_steps=max_steps,
        )

    def run_inference(self, rule_timekey=None):
        return self._infer.run(rule_timekey=rule_timekey)
