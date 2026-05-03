import argparse
import random

import numpy as np

from ApproPO.args import appropo_args
from ApproPO.control_solver import run
from ApproPO.envs.pid_control_env import PIControlEnv
from ApproPO.pid_oracle import PIGainPolicy, PIOracle


def add_pi_args(parser):
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pi_horizon", type=int, default=100)
    parser.add_argument("--pi_dt", type=float, default=0.1)
    parser.add_argument("--pi_tau", type=float, default=2.0)
    parser.add_argument("--pi_u_limit", type=float, default=2.0)
    parser.add_argument("--pi_effort_weight", type=float, default=1.0)
    parser.add_argument("--response_tolerance", type=float, default=0.05)
    parser.add_argument("--min_reward", type=float, default=-13.5)
    parser.add_argument("--max_response_time", type=float, default=3.0)
    parser.add_argument("--initial_reward_weight", type=float, default=1.0)
    parser.add_argument("--initial_response_weight", type=float, default=0.0)
    parser.add_argument("--constraint_weight", type=float, default=20.0)
    parser.add_argument("--constraint_ema", type=float, default=0.5)
    parser.add_argument("--pi_search_step", type=float, default=0.35)
    parser.add_argument("--pi_elite_frac", type=float, default=0.25)
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--plot_output", type=str, default="results/pi_appropo_responses.png")
    parser.add_argument("--plot_max_curves", type=int, default=6)


parser = argparse.ArgumentParser()
appropo_args(parser)
add_pi_args(parser)
args = parser.parse_args()

seed = args.seed if args.seed is not None else random.randint(0, 100000)
np.random.seed(seed)
random.seed(seed)
print(f"Seed: {seed}")


def project_measurements(measurements):
    reward, neg_response_time = measurements
    return np.array(
        [
            max(reward, args.min_reward),
            max(neg_response_time, -args.max_response_time),
        ]
    )


def main():
    env = PIControlEnv(
        horizon=args.pi_horizon,
        dt=args.pi_dt,
        tau=args.pi_tau,
        u_limit=args.pi_u_limit,
        effort_weight=args.pi_effort_weight,
        response_tolerance=args.response_tolerance,
        seed=seed,
    )
    policy = PIGainPolicy(seed=seed)
    oracle = PIOracle(env=env, theta=np.array([-1.0, 0.0]), policy=policy, args=args)
    run(proj=project_measurements, oracle=oracle, args=args)


if __name__ == "__main__":
    main()
