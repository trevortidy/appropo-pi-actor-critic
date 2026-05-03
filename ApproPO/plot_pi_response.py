import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ApproPO.envs.pid_control_env import PIControlEnv


def parse_gains(text):
    gain_sets = []
    for item in text.split(";"):
        kp, ki = item.split(",")
        gain_sets.append((float(kp), float(ki)))
    return gain_sets


def plot_responses(env, gain_sets, output):
    plt.figure(figsize=(7, 4.5))
    setpoint_plotted = False

    for kp, ki in gain_sets:
        response = env.response((kp, ki))
        if not setpoint_plotted:
            plt.plot(
                response["time"],
                response["setpoint"],
                "k--",
                linewidth=1.2,
                label="setpoint",
            )
            setpoint_plotted = True
        label = f"Kp={kp:g}, Ki={ki:g}"
        plt.plot(response["time"], response["output"], linewidth=1.8, label=label)

    plt.xlabel("time")
    plt.ylabel("process output")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gains", default="1,0.3;2,0.8;5,1.5")
    parser.add_argument("--output", default="results/pi_response.png")
    parser.add_argument("--pi_horizon", type=int, default=100)
    parser.add_argument("--pi_dt", type=float, default=0.1)
    parser.add_argument("--pi_tau", type=float, default=2.0)
    parser.add_argument("--pi_u_limit", type=float, default=2.0)
    args = parser.parse_args()

    env = PIControlEnv(
        horizon=args.pi_horizon,
        dt=args.pi_dt,
        tau=args.pi_tau,
        u_limit=args.pi_u_limit,
    )
    plot_responses(env, parse_gains(args.gains), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
