import numpy as np


class PIGainPolicy:
    def __init__(self, gain_high=None, seed=None):
        if gain_high is None:
            gain_high = [8.0, 3.0]
        self.gain_high = np.asarray(gain_high, dtype=np.float64)
        self.loc = np.array([0.0, 0.0], dtype=np.float64)
        self.std = np.array([0.8, 0.8], dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def sample(self):
        raw = self.rng.normal(self.loc, self.std)
        gains = self.gain_high / (1.0 + np.exp(-raw))
        return raw, gains

    def mean_gains(self):
        return self.gain_high / (1.0 + np.exp(-self.loc))

    def fit_elite(self, raw_samples, scores, elite_frac=0.25, step_size=0.35):
        n_elite = max(1, int(np.ceil(elite_frac * len(scores))))
        elite = raw_samples[np.argsort(scores)[:n_elite]]
        self.loc = (1.0 - step_size) * self.loc + step_size * elite.mean(axis=0)
        self.std = (1.0 - step_size) * self.std + step_size * np.maximum(
            elite.std(axis=0), 0.05
        )

    def state_dict(self):
        return {
            "loc": self.loc.copy(),
            "std": self.std.copy(),
        }

    def load_state_dict(self, state):
        self.loc = np.asarray(state["loc"], dtype=np.float64).copy()
        self.std = np.asarray(state["std"], dtype=np.float64).copy()

    def clone(self, seed=None):
        policy = PIGainPolicy(self.gain_high.copy(), seed=seed)
        policy.load_state_dict(self.state_dict())
        return policy


class PIOracle:
    """Inner oracle that searches for PI gains under the current scalarization."""

    def __init__(self, env=None, theta=None, policy=None, args=None):
        self.env = env
        self.theta = theta
        self.policy = policy or PIGainPolicy()
        self.elite_frac = args.pi_elite_frac
        self.step_size = args.pi_search_step
        self.restarts = args.pi_restarts
        self.grid_size = args.pi_grid_size
        self.rng = np.random.default_rng(args.seed)

    def _score(self, measurements):
        theta = np.asarray(self.theta, dtype=np.float64)
        if theta.size == np.asarray(measurements).size + 1:
            return float(np.dot(theta, np.append(measurements, self.mx_size)))
        return float(np.dot(theta, measurements))

    def _learn_with_policy(self, policy, n_traj=24, n_iter=5):
        original_policy = self.policy
        self.policy = policy
        final_measurements = None
        final_stats = None

        for _ in range(n_iter):
            raw_samples = []
            scores = []
            measurements_batch = []
            stats_batch = []

            for _ in range(n_traj):
                raw, gains = self.policy.sample()
                measurements, stats = self.env.rollout(gains)
                raw_samples.append(raw)
                scores.append(self._score(measurements))
                measurements_batch.append(measurements)
                stats_batch.append(stats)

            self.policy.fit_elite(
                np.asarray(raw_samples),
                np.asarray(scores),
                elite_frac=self.elite_frac,
                step_size=self.step_size,
            )
            final_measurements = np.average(measurements_batch, axis=0)
            final_stats = stats_batch

        self.policy = original_policy
        return final_measurements, final_stats

    def learn_policy(self, n_traj=24, n_iter=5):
        if self.grid_size > 1:
            return self._learn_policy_grid()

        candidates = [
            self.policy.clone(seed=int(self.rng.integers(0, 2**31 - 1)))
        ]
        for _ in range(max(0, self.restarts - 1)):
            candidates.append(
                PIGainPolicy(seed=int(self.rng.integers(0, 2**31 - 1)))
            )

        best_score = float("inf")
        best_policy = None
        best_measurements = None
        best_stats = None
        for candidate in candidates:
            measurements, stats = self._learn_with_policy(candidate, n_traj, n_iter)
            score = self._score(measurements)
            if score < best_score:
                best_score = score
                best_policy = candidate
                best_measurements = measurements
                best_stats = stats

        self.policy.load_state_dict(best_policy.state_dict())
        return best_measurements, best_stats

    def _learn_policy_grid(self):
        best_score = float("inf")
        best_gains = None
        best_measurements = None
        best_stats = None
        kp_values = np.linspace(0.0, self.policy.gain_high[0], self.grid_size)
        ki_values = np.linspace(0.0, self.policy.gain_high[1], self.grid_size)

        for kp in kp_values:
            for ki in ki_values:
                gains = np.array([kp, ki], dtype=np.float64)
                measurements, stats = self.env.rollout(gains)
                score = self._score(measurements)
                if score < best_score:
                    best_score = score
                    best_gains = gains
                    best_measurements = measurements
                    best_stats = [stats]

        eps = 1e-9
        clipped = np.clip(best_gains / self.policy.gain_high, eps, 1.0 - eps)
        self.policy.loc = np.log(clipped / (1.0 - clipped))
        self.policy.std = np.full_like(self.policy.std, 0.05)
        return best_measurements, best_stats

    def evaluate_mean_policy(self, n_traj=12):
        gains = self.policy.mean_gains()
        measurements_batch = []
        stats_batch = []
        for _ in range(n_traj):
            measurements, stats = self.env.rollout(gains)
            measurements_batch.append(measurements)
            stats_batch.append(stats)
        return np.average(measurements_batch, axis=0), stats_batch
