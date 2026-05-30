"""Unit checks for separated observation/reward policy modules."""
import numpy as np

from biz.services.rl.config.policy import load_env_policy
from biz.services.rl.env.observation import build_observation_from_env, observation_dim
from biz.services.rl.env.reward import RewardConfig, RewardStepContext, compute_reward
from biz.services.rl.env.scheduler_env import SchedulerEnv
from biz.services.rl.validation.fixtures import TestDataLoader


def test_observation_dim_matches_env():
    data = TestDataLoader().load_for_env("benchmark_dataset")
    env = SchedulerEnv(data, max_prods=3, max_procs=3, max_models=3, max_steps=4)
    obs, _ = env.reset()
    expected = observation_dim(env.num_prods, env.num_procs, env.num_models)
    assert obs.shape == (expected,)
    assert obs.shape == env.observation_space.shape
    assert np.allclose(obs, build_observation_from_env(env, env.observation_norm))


def test_reward_from_yaml_defaults():
    policy = load_env_policy()
    plan = np.array([[100.0]])
    produced = np.array([[100.0]])
    ctx = RewardStepContext(
        step_production=10.0,
        plan=plan,
        produced=produced,
        num_transfers=2,
        prev_guidance_gap=0.0,
        next_guidance_gap=0.0,
        guidance_target_eqp=np.zeros((1, 1, 1)),
        terminated=True,
    )
    r = compute_reward(ctx, policy.reward)
    # production 10/100 + terminal 1.5 (100%) - 2 * transfer penalty
    expected = 0.1 + policy.reward.terminal_bonus_99 - 2 * policy.reward.transfer_penalty_per_unit
    assert abs(r - expected) < 1e-6


if __name__ == "__main__":
    test_observation_dim_matches_env()
    test_reward_from_yaml_defaults()
    print("test_env_policy: OK")
