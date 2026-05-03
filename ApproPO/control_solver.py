import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _avg(stats, key):
    return float(np.average([item[key] for item in stats]))


def _plot_epoch_responses(env, rows, args):
    if not rows:
        return

    stride = max(1, int(np.ceil(len(rows) / args.plot_max_curves)))
    selected = rows[::stride]
    if selected[-1]["epoch"] != rows[-1]["epoch"]:
        selected.append(rows[-1])

    plt.figure(figsize=(7, 4.5))
    setpoint_plotted = False
    for row in selected:
        response = env.response((row["kp"], row["ki"]))
        if not setpoint_plotted:
            plt.plot(
                response["time"],
                response["setpoint"],
                "k--",
                linewidth=1.2,
                label="setpoint",
            )
            setpoint_plotted = True
        plt.plot(
            response["time"],
            response["output"],
            linewidth=1.6,
            label=f"epoch {row['epoch']} (Kp={row['kp']:.2f}, Ki={row['ki']:.2f})",
        )

    plt.xlabel("time")
    plt.ylabel("process output")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.plot_output) or ".", exist_ok=True)
    plt.savefig(args.plot_output, dpi=160)
    plt.close()
    print(f"Wrote {args.plot_output}")


def run(proj=None, oracle=None, args=None):
    theta = -np.array(
        [args.initial_reward_weight, args.initial_response_weight],
        dtype=np.float64,
    )
    violation_ema = np.zeros(2, dtype=np.float64)
    rows = []

    for epoch in range(args.num_epochs):
        oracle.theta = theta
        oracle.learn_policy(n_traj=args.rl_traj, n_iter=args.rl_iter)
        measurements, stats = oracle.evaluate_mean_policy(n_traj=args.check_traj)

        projected = proj(measurements)
        violation = np.maximum(projected - measurements, 0.0)
        scale = np.maximum(np.abs(projected), 1e-6)
        normalized_violation = violation / scale
        violation_ema = (
            args.constraint_ema * violation_ema
            + (1.0 - args.constraint_ema) * normalized_violation
        )
        weights = (
            np.array(
                [args.initial_reward_weight, args.initial_response_weight],
                dtype=np.float64,
            )
            + args.constraint_weight * violation_ema
        )
        next_theta = -weights

        reward = measurements[0]
        response_time = -measurements[1]
        row = {
            "epoch": epoch,
            "reward": reward,
            "response_time": response_time,
            "reward_violation": violation[0],
            "response_violation": violation[1],
            "reward_weight": weights[0],
            "response_weight": weights[1],
            "kp": _avg(stats, "kp"),
            "ki": _avg(stats, "ki"),
        }
        rows.append(row)

        print(
            f"Epoch {epoch}: reward={reward:.3f}, "
            f"response_time={response_time:.2f}s, "
            f"reward_violation={violation[0]:.3f}, "
            f"response_violation={violation[1]:.2f}, "
            f"kp={row['kp']:.3f}, ki={row['ki']:.3f}, "
            f"weights=({weights[0]:.2f}, {weights[1]:.2f})"
        )

        theta = next_theta

    if args.print:
        os.makedirs(args.output, exist_ok=True)
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        path = f"{args.output}/pi_feasibility_{args.name}_{date}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Wrote {path}")

    if args.plot:
        _plot_epoch_responses(oracle.env, rows, args)
