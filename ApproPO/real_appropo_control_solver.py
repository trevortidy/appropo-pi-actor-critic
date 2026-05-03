from collections import namedtuple
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CacheItem = namedtuple("CacheItem", ["measurement", "policy_state", "gains"])


def _is_feasible(measurement, proj, atol=1e-8):
    return np.linalg.norm(measurement - proj(measurement.copy())) <= atol


def _state_from_oracle(oracle):
    return oracle.policy.state_dict()


def _load_state(oracle, state):
    oracle.policy.load_state_dict(state)


def _gains_from_oracle(oracle):
    return oracle.policy.mean_gains()


def _scalar_value(theta, measurement, args):
    return float(np.dot(theta, np.append(measurement, args.mx_size)))


def _make_policy_state_from_gains(policy, gains):
    if hasattr(policy, "state_from_gains"):
        return policy.state_from_gains(gains)
    eps = 1e-9
    gains = np.asarray(gains, dtype=np.float64)
    clipped = np.clip(gains / policy.gain_high, eps, 1.0 - eps)
    return {
        "loc": np.log(clipped / (1.0 - clipped)),
        "std": np.full_like(policy.std, 0.05),
    }


def _average_response(env, gains_components):
    responses = [env.response(gains) for gains in gains_components]
    output = np.average([response["output"] for response in responses], axis=0)
    return {
        "time": responses[0]["time"],
        "setpoint": responses[0]["setpoint"],
        "output": output,
    }


def _plot_responses(env, rows, args):
    if not args.plot or not rows:
        return
    stride = max(1, int(np.ceil(len(rows) / args.plot_max_curves)))
    selected = rows[::stride]
    if selected[-1]["epoch"] != rows[-1]["epoch"]:
        selected.append(rows[-1])

    plt.figure(figsize=(7, 4.5))
    setpoint_plotted = False
    for row in selected:
        response = _average_response(env, row["average_policy"])
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
            label=(
                f"avg epoch {row['epoch']} "
                f"(reward={row['avg_reward']:.2f}, t={row['avg_response_time']:.2f}s)"
            ),
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


def run(proj_oracle=None, oracle_generator=None, args=None):
    theta = proj_oracle.get_theta()
    cache = []
    history = []
    policy_history = []
    rows = []
    best_feasible = None

    init_oracle = oracle_generator()
    for _ in range(args.cache_size):
        _, gains = init_oracle.policy.sample()
        measurement, _ = init_oracle.env.rollout(gains)
        cache.append(
            CacheItem(
                measurement.copy(),
                _make_policy_state_from_gains(init_oracle.policy, gains),
                gains.copy(),
            )
        )

    for epoch in range(args.num_epochs):
        oracle = oracle_generator()
        oracle.theta = theta
        oracle.mx_size = args.mx_size

        positive_cached_item = None
        positive_cached_value = float("inf")
        warm_start_item = None
        warm_start_value = float("inf")
        for item in cache:
            value = _scalar_value(theta, item.measurement, args)
            if value < warm_start_value:
                warm_start_value = value
                warm_start_item = item
            if value <= args.positive_response_epsilon and value < positive_cached_value:
                positive_cached_value = value
                positive_cached_item = item

        used_cache = (
            positive_cached_item is not None
            and positive_cached_value <= args.cache_tolerance
        )
        if used_cache:
            measurement = positive_cached_item.measurement.copy()
            _load_state(oracle, positive_cached_item.policy_state)
            gains = positive_cached_item.gains.copy()
        else:
            if warm_start_item is not None:
                _load_state(oracle, warm_start_item.policy_state)
            oracle.learn_policy(n_traj=args.rl_traj, n_iter=args.rl_iter)
            measurement, _ = oracle.evaluate_mean_policy(n_traj=args.check_traj)
            gains = _gains_from_oracle(oracle)
            cache.append(
                CacheItem(measurement.copy(), _state_from_oracle(oracle), gains.copy())
            )

        scalar_value = _scalar_value(theta, measurement, args)
        positive_response = scalar_value <= args.positive_response_epsilon
        proj_oracle.update(measurement.copy())
        theta = proj_oracle.get_theta()
        history.append(measurement.copy())
        policy_history.append(gains.copy())

        avg_measurement = np.average(np.stack(history), axis=0)
        dist_current = np.linalg.norm(measurement - proj_oracle.proj(measurement.copy()))
        dist_average = np.linalg.norm(
            avg_measurement - proj_oracle.proj(avg_measurement.copy())
        )
        feasible_current = _is_feasible(
            measurement, proj_oracle.proj, atol=args.feasibility_tolerance
        )
        feasible_average = _is_feasible(
            avg_measurement, proj_oracle.proj, atol=args.feasibility_tolerance
        )
        if feasible_current and best_feasible is None:
            best_feasible = CacheItem(
                measurement.copy(), _state_from_oracle(oracle), gains.copy()
            )

        reward = measurement[0]
        response_time = -measurement[1]
        avg_reward = avg_measurement[0]
        avg_response_time = -avg_measurement[1]
        rows.append(
            {
                "epoch": epoch,
                "reward": reward,
                "response_time": response_time,
                "avg_reward": avg_reward,
                "avg_response_time": avg_response_time,
                "kp": float(gains[0]),
                "ki": float(gains[1]),
                "average_policy": [g.copy() for g in policy_history],
                "dist_current": dist_current,
                "dist_average": dist_average,
                "cache": used_cache,
                "feasible_current": feasible_current,
                "feasible_average": feasible_average,
                "scalar_value": scalar_value,
                "positive_response": positive_response,
            }
        )
        print(
            f"Epoch {epoch}: reward={reward:.3f}, "
            f"response_time={response_time:.2f}s, "
            f"dist_current={dist_current:.3f}, "
            f"dist_average={dist_average:.3f}, "
            f"kp={gains[0]:.3f}, ki={gains[1]:.3f}, "
            f"avg_reward={avg_reward:.3f}, "
            f"avg_response_time={avg_response_time:.2f}s, "
            f"scalar_value={scalar_value:.3f}, "
            f"positive_response={positive_response}, "
            f"theta={theta[:-1]}, "
            f"cache={used_cache}, "
            f"feasible_current={feasible_current}, "
            f"feasible_average={feasible_average}"
        )

        if feasible_average and args.stop_on_feasible:
            break

    if best_feasible is not None:
        reward = best_feasible.measurement[0]
        response_time = -best_feasible.measurement[1]
        print(
            f"Best feasible: reward={reward:.3f}, "
            f"response_time={response_time:.2f}s"
        )

    avg_measurement = np.average(np.stack(history), axis=0)
    avg_reward = avg_measurement[0]
    avg_response_time = -avg_measurement[1]
    avg_feasible = _is_feasible(
        avg_measurement, proj_oracle.proj, atol=args.feasibility_tolerance
    )
    print(
        f"Average policy: reward={avg_reward:.3f}, "
        f"response_time={avg_response_time:.2f}s, "
        f"feasible={avg_feasible}"
    )
    print("Average policy components:")
    for idx, gains in enumerate(policy_history):
        print(f"  {idx}: kp={gains[0]:.3f}, ki={gains[1]:.3f}")

    _plot_responses(oracle.env, rows, args)

    return {
        "cache": cache,
        "history": history,
        "average_policy": policy_history,
        "average_measurement": np.average(np.stack(history), axis=0) if history else None,
        "best_feasible": best_feasible,
    }
