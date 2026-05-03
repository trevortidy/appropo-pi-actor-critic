import argparse
import random

import numpy as np

from ApproPO.args import appropo_args
from ApproPO.envs.pid_control_env import PIControlEnv
from ApproPO.pi_actor_critic_oracle import (
    PIActorCriticEnv,
    PIActorCriticOracle,
    PIActorCriticPolicy,
)
from ApproPO.projection_oracle import ProjectionOracle
from ApproPO.real_appropo_control_solver import run


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
    parser.add_argument("--pi_search_step", type=float, default=0.35)
    parser.add_argument("--pi_elite_frac", type=float, default=0.25)
    parser.add_argument("--pi_restarts", type=int, default=4)
    parser.add_argument("--pi_grid_size", type=int, default=81)
    parser.add_argument("--pi_ac_gain_step", type=float, default=0.08)
    parser.add_argument("--cache_tolerance", type=float, default=0.0)
    parser.add_argument("--positive_response_epsilon", type=float, default=0.0)
    parser.add_argument("--feasibility_tolerance", type=float, default=0.1)
    parser.add_argument("--stop_on_feasible", default=False, action="store_true")
    parser.add_argument("--olo_momentum", type=float, default=0.9)
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--plot_output", type=str, default="results/pi_real_appropo.png")
    parser.add_argument("--plot_max_curves", type=int, default=6)


parser = argparse.ArgumentParser()
appropo_args(parser)
add_pi_args(parser)
args = parser.parse_args()
args.diversity = True
args.init_variable = "None"

seed = args.seed if args.seed is not None else random.randint(0, 100000)
np.random.seed(seed)
random.seed(seed)
print(f"Seed: {seed}")


def project_measurements(measurement):
    reward, neg_response_time = measurement
    return np.array(
        [
            max(reward, args.min_reward),
            max(neg_response_time, -args.max_response_time),
        ],
        dtype=np.float64,
    )


def main():
    theta0 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    control_env = PIControlEnv(
        horizon=args.pi_horizon,
        dt=args.pi_dt,
        tau=args.pi_tau,
        u_limit=args.pi_u_limit,
        effort_weight=args.pi_effort_weight,
        response_tolerance=args.response_tolerance,
        seed=seed,
    )
    env = PIActorCriticEnv(
        control_env=control_env,
        theta=theta0,
        gain_step=args.pi_ac_gain_step,
        min_reward=args.min_reward,
        max_response_time=args.max_response_time,
        mx_size=args.mx_size,
        seed=seed,
    )
    policy = PIActorCriticPolicy(env=env, seed=seed, device=args.device)

    def oracle_generator():
        oracle = PIActorCriticOracle(
            env=env,
            theta=theta0,
            policy=policy,
            args=args,
        )
        oracle.mx_size = args.mx_size
        return oracle

    proj_oracle = ProjectionOracle(
        dim=2,
        proj=project_measurements,
        args=args,
    )
    proj_oracle.olo.thetas[-1] = theta0.copy()
    proj_oracle.olo.mass = args.olo_momentum
    run(proj_oracle=proj_oracle, oracle_generator=oracle_generator, args=args)


if __name__ == "__main__":
    main()
