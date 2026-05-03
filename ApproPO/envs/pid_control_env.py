import numpy as np


class PIControlEnv:
    """Minimal PI tuning benchmark.

    The agent chooses Kp and Ki once per episode. The environment simulates a
    first-order plant tracking a unit step and returns only the two quantities
    APPROPO cares about: reward and response time.
    """

    def __init__(
        self,
        horizon=100,
        dt=0.1,
        tau=2.0,
        gain=1.0,
        u_limit=2.0,
        effort_weight=0.08,
        response_tolerance=0.05,
        seed=None,
    ):
        self.horizon = horizon
        self.dt = dt
        self.tau = tau
        self.gain = gain
        self.u_limit = u_limit
        self.effort_weight = effort_weight
        self.response_tolerance = response_tolerance
        self.rng = np.random.default_rng(seed)

    def rollout(self, gains):
        response = self.response(gains)
        total_cost = response["tracking_cost"] + self.effort_weight * response["effort_cost"]
        reward = -total_cost
        measurements = np.array([reward, -response["response_time"]], dtype=np.float64)
        stats = {
            "reward": reward,
            "response_time": response["response_time"],
            "kp": response["kp"],
            "ki": response["ki"],
        }
        return measurements, stats

    def response(self, gains):
        kp, ki = np.maximum(np.asarray(gains, dtype=np.float64), 0.0)
        y = 0.0
        integral = 0.0
        tracking_cost = 0.0
        effort_cost = 0.0
        response_time = self.horizon * self.dt
        times = []
        outputs = []
        setpoints = []

        for t in range(self.horizon):
            setpoint = 1.0
            error = setpoint - y
            integral += error * self.dt
            u = np.clip(kp * error + ki * integral, -self.u_limit, self.u_limit)

            y += self.dt * (-y + self.gain * u) / self.tau

            tracking_cost += abs(error) * self.dt
            effort_cost += (u ** 2) * self.dt
            if response_time == self.horizon * self.dt and abs(error) <= self.response_tolerance:
                response_time = t * self.dt
            times.append(t * self.dt)
            outputs.append(y)
            setpoints.append(setpoint)

        return {
            "time": np.asarray(times),
            "output": np.asarray(outputs),
            "setpoint": np.asarray(setpoints),
            "tracking_cost": tracking_cost,
            "effort_cost": effort_cost,
            "response_time": response_time,
            "kp": float(kp),
            "ki": float(ki),
        }
