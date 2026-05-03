import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--pi_ac_gain_step", type=float, default=0.08)
    parser.add_argument("--cache_tolerance", type=float, default=0.0)
    parser.add_argument("--positive_response_epsilon", type=float, default=0.0)
    parser.add_argument("--feasibility_tolerance", type=float, default=0.1)
    parser.add_argument("--stop_on_feasible", default=False, action="store_true")
    parser.add_argument("--olo_momentum", type=float, default=0.9)
    parser.add_argument("--component_stride", type=int, default=1)
    parser.add_argument(
        "--summary_output",
        type=str,
        default="results/pi_mixture_summary.png",
    )


def project_measurements(measurement, args):
    reward, neg_response_time = measurement
    return np.array(
        [
            max(reward, args.min_reward),
            max(neg_response_time, -args.max_response_time),
        ],
        dtype=np.float64,
    )


def is_feasible_measurement(measurement, args):
    reward, neg_response_time = measurement
    return reward >= args.min_reward and -neg_response_time <= args.max_response_time


def build_experiment(args, seed):
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
        proj=lambda measurement: project_measurements(measurement, args),
        args=args,
    )
    proj_oracle.olo.thetas[-1] = theta0.copy()
    proj_oracle.olo.mass = args.olo_momentum
    return env, proj_oracle, oracle_generator


def component_color(measurement, args):
    if is_feasible_measurement(measurement, args):
        return "#2ca02c"
    if measurement[0] < args.min_reward:
        return "#d62728"
    return "#ff7f0e"


def plot_summary(env, result, args):
    policies = result["average_policy"]
    history = result["history"]
    avg_measurements = [
        np.average(np.stack(history[: idx + 1]), axis=0)
        for idx in range(len(history))
    ]
    final_average = avg_measurements[-1]
    final_feasible = is_feasible_measurement(final_average, args)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_response, ax_measurement = axes

    selected_indices = range(0, len(policies), max(1, args.component_stride))
    tolerance_low = 1.0 - args.response_tolerance
    tolerance_high = 1.0 + args.response_tolerance

    ax_response.axhspan(
        tolerance_low,
        tolerance_high,
        color="#2ca02c",
        alpha=0.10,
        label="response tolerance",
    )
    ax_response.axvline(
        args.max_response_time,
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        label="time target",
    )

    setpoint_plotted = False
    for idx in selected_indices:
        gains = policies[idx]
        measurement, _ = env.rollout(gains)
        response = env.response(gains)
        if not setpoint_plotted:
            ax_response.plot(
                response["time"],
                response["setpoint"],
                "k--",
                linewidth=1.2,
                label="setpoint",
            )
            setpoint_plotted = True
        ax_response.plot(
            response["time"],
            response["output"],
            color=component_color(measurement, args),
            linewidth=0.9,
            alpha=0.28,
        )

    ax_response.set_title("Mixture component responses")
    ax_response.set_xlabel("time")
    ax_response.set_ylabel("process output")
    ax_response.grid(True, alpha=0.3)
    ax_response.legend(fontsize=8, loc="lower right")

    rewards = np.array([measurement[0] for measurement in history])
    response_times = np.array([-measurement[1] for measurement in history])
    avg_rewards = np.array([measurement[0] for measurement in avg_measurements])
    avg_response_times = np.array([-measurement[1] for measurement in avg_measurements])

    x_min = min(rewards.min(), avg_rewards.min(), args.min_reward) - 0.5
    x_max = max(rewards.max(), avg_rewards.max(), args.min_reward) + 0.5
    y_min = 0.0
    y_max = max(response_times.max(), avg_response_times.max(), args.max_response_time) + 0.5

    ax_measurement.axvspan(
        args.min_reward,
        x_max,
        ymin=0.0,
        ymax=(args.max_response_time - y_min) / (y_max - y_min),
        color="#2ca02c",
        alpha=0.10,
        label="target set",
    )
    ax_measurement.axvline(args.min_reward, color="#444444", linestyle="--", linewidth=1.1)
    ax_measurement.axhline(args.max_response_time, color="#444444", linestyle="--", linewidth=1.1)
    ax_measurement.scatter(
        rewards,
        response_times,
        s=22,
        color="#1f77b4",
        alpha=0.45,
        label="returned policies",
    )
    ax_measurement.plot(
        avg_rewards,
        avg_response_times,
        color="#111111",
        linewidth=1.6,
        label="running average",
    )
    ax_measurement.scatter(
        [final_average[0]],
        [-final_average[1]],
        marker="*",
        s=170,
        color="#d62728" if not final_feasible else "#2ca02c",
        edgecolor="#111111",
        linewidth=0.7,
        label="final average",
        zorder=5,
    )
    ax_measurement.set_xlim(x_min, x_max)
    ax_measurement.set_ylim(y_min, y_max)
    ax_measurement.set_title("APPROPO measurement space")
    ax_measurement.set_xlabel("reward")
    ax_measurement.set_ylabel("response time")
    ax_measurement.grid(True, alpha=0.3)
    ax_measurement.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        (
            f"Final average: reward={final_average[0]:.3f}, "
            f"response_time={-final_average[1]:.2f}s, feasible={final_feasible}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    fig.savefig(args.summary_output, dpi=170)
    plt.close(fig)
    print(f"Wrote {args.summary_output}")


def main():
    parser = argparse.ArgumentParser()
    appropo_args(parser)
    add_pi_args(parser)
    args = parser.parse_args()
    args.diversity = True
    args.init_variable = "None"
    args.plot = False

    seed = args.seed if args.seed is not None else random.randint(0, 100000)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")

    env, proj_oracle, oracle_generator = build_experiment(args, seed)
    result = run(
        proj_oracle=proj_oracle,
        oracle_generator=oracle_generator,
        args=args,
    )
    plot_summary(env, result, args)


if __name__ == "__main__":
    main()
