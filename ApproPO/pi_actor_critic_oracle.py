from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical

from ApproPO.nets import MLP


SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class PIActorCriticPolicy:
    def __init__(self, env, gain_high=None, seed=None, device="cpu"):
        if gain_high is None:
            gain_high = [8.0, 3.0]
        self.gain_high = np.asarray(gain_high, dtype=np.float64)
        self.gains = 0.5 * self.gain_high
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.net = MLP(env).to(device)

    def sample(self):
        gains = self.rng.uniform(np.zeros_like(self.gain_high), self.gain_high)
        return gains.copy(), gains.copy()

    def mean_gains(self):
        return self.gains.copy()

    def set_gains(self, gains):
        self.gains = np.clip(np.asarray(gains, dtype=np.float64), 0.0, self.gain_high)

    def state_from_gains(self, gains):
        return {
            "gains": np.clip(np.asarray(gains, dtype=np.float64), 0.0, self.gain_high),
            "net": {
                key: value.detach().clone()
                for key, value in self.net.state_dict().items()
            },
        }

    def state_dict(self):
        return self.state_from_gains(self.gains)

    def load_state_dict(self, state):
        self.set_gains(state["gains"])
        if "net" in state:
            self.net.load_state_dict(state["net"])


class PIActorCriticEnv:
    """Gym-style PI gain update environment for the APPROPO inner oracle."""

    def __init__(
        self,
        control_env=None,
        theta=None,
        gain_high=None,
        gain_step=0.08,
        min_reward=-13.5,
        max_response_time=3.0,
        mx_size=20,
        seed=None,
    ):
        self.control_env = control_env
        self.theta = theta
        self.gain_high = np.asarray(gain_high or [8.0, 3.0], dtype=np.float64)
        self.gain_delta = self.gain_high * gain_step
        self.min_reward = min_reward
        self.max_response_time = max_response_time
        self.mx_size = mx_size
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        self.gains = 0.5 * self.gain_high
        self.measurements = None
        self.stats = None

    def reset(self, gains=None):
        if gains is None:
            self.gains = self.rng.uniform(np.zeros_like(self.gain_high), self.gain_high)
        else:
            self.gains = np.clip(np.asarray(gains, dtype=np.float64), 0.0, self.gain_high)
        self.measurements, self.stats = self.control_env.rollout(self.gains)
        return self._state(), {}

    def step(self, action):
        if action == 0:
            self.gains[0] += self.gain_delta[0]
        elif action == 1:
            self.gains[0] -= self.gain_delta[0]
        elif action == 2:
            self.gains[1] += self.gain_delta[1]
        elif action == 3:
            self.gains[1] -= self.gain_delta[1]
        self.gains = np.clip(self.gains, 0.0, self.gain_high)
        self.measurements, self.stats = self.control_env.rollout(self.gains)
        # APPROPO's positive-response oracle minimizes the scalar value, while
        # the actor-critic update maximizes reward.
        reward = -self.scalar_score(self.measurements)
        return self._state(), reward, False, False, self.info()

    def rollout(self, gains):
        return self.control_env.rollout(gains)

    def response(self, gains):
        return self.control_env.response(gains)

    def scalar_score(self, measurements):
        theta = np.asarray(self.theta, dtype=np.float64)
        if theta.size == np.asarray(measurements).size + 1:
            return float(np.dot(theta, np.append(measurements, self.mx_size)))
        return float(np.dot(theta, measurements))

    def info(self):
        info = dict(self.stats)
        info["measurements"] = self.measurements.copy()
        info["gains"] = self.gains.copy()
        info["scalar_score"] = self.scalar_score(self.measurements)
        return info

    def _state(self):
        reward = float(self.measurements[0])
        response_time = float(-self.measurements[1])
        reward_scale = max(abs(self.min_reward), 1.0)
        time_scale = max(self.control_env.horizon * self.control_env.dt, 1.0)
        reward_violation = max(0.0, self.min_reward - reward) / reward_scale
        time_violation = max(0.0, response_time - self.max_response_time) / time_scale
        return np.array(
            [
                self.gains[0] / self.gain_high[0],
                self.gains[1] / self.gain_high[1],
                reward / reward_scale,
                response_time / time_scale,
                reward_violation,
                time_violation,
            ],
            dtype=np.float32,
        )


class PIActorCriticOracle:
    """A2C positive-response oracle for PI gain tuning."""

    def __init__(self, env=None, theta=None, policy=None, args=None):
        self.env = env
        self.theta = theta
        self.mx_size = args.mx_size
        self.gamma = args.gamma
        self.lr = args.rl_lr
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.device = args.device
        self.eps = np.finfo(np.float32).eps.item()
        self.policy = policy or PIActorCriticPolicy(
            env=env,
            seed=args.seed,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(self.policy.net.parameters(), lr=self.lr)
        self.saved_actions = []
        self.rewards = []
        self.entropies = []

    def reset(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.entropies[:]

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_scores, state_value = self.policy.net(state)
        distribution = Categorical(logits=-action_scores)
        action = distribution.sample()
        self.saved_actions.append(SavedAction(distribution.log_prob(action), state_value))
        self.entropies.append(distribution.entropy())
        return action.item()

    def finish_episode(self):
        returns = []
        reward_to_go = 0.0
        for reward in self.rewards[::-1]:
            reward_to_go = reward + self.gamma * reward_to_go
            returns.insert(0, reward_to_go)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        policy_losses = []
        value_losses = []
        for (log_prob, value), reward_to_go in zip(self.saved_actions, returns):
            advantage = reward_to_go.item() - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value.view(-1), reward_to_go.view(-1)))

        self.optimizer.zero_grad()
        loss = (
            torch.stack(policy_losses).mean()
            + self.value_coef * torch.stack(value_losses).mean()
            - self.entropy_coef * torch.stack(self.entropies).mean()
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.net.parameters(), 0.5)
        self.optimizer.step()
        self.reset()

    def learn_policy(self, n_traj=20, n_iter=300):
        self.env.theta = self.theta
        self.env.mx_size = self.mx_size
        best_score = float("inf")
        best_measurements = None
        best_gains = None
        stats_batch = []

        for _ in range(n_traj):
            state, _ = self.env.reset(gains=self.policy.mean_gains())
            self.reset()
            final_info = None

            for _ in range(n_iter):
                action = self.select_action(state)
                state, reward, _, _, info = self.env.step(action)
                self.rewards.append(reward)
                final_info = info
                if info["scalar_score"] < best_score:
                    best_score = info["scalar_score"]
                    best_measurements = info["measurements"].copy()
                    best_gains = info["gains"].copy()

            self.finish_episode()
            if final_info is not None:
                stats_batch.append(final_info)

        if best_gains is not None:
            self.policy.set_gains(best_gains)
        return best_measurements, stats_batch

    def evaluate_mean_policy(self, n_traj=12):
        gains = self.policy.mean_gains()
        measurements_batch = []
        stats_batch = []
        for _ in range(n_traj):
            measurements, stats = self.env.control_env.rollout(gains)
            measurements_batch.append(measurements)
            stats_batch.append(stats)
        return np.average(measurements_batch, axis=0), stats_batch
