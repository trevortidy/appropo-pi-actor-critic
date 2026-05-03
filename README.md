# Reinforcement Learning with Convex Constraints

PyTorch implementation of the paper:

[Reinforcement Learning with Convex Constraints](https://papers.nips.cc/paper/9556-reinforcement-learning-with-convex-constraints.pdf)\
Sobhan Miryoosefi, Kianté Brantley, Hal Daumé III, Miroslav Dudik, Robert Schapire\
NeurIPS 2019 

```bash
python3 -m pip install -e .
```

## Note on Mars Rover Reproduction

This repository keeps the original Mars Rover environment and scripts, but
`ApproPO/solver.py` has been modified for the PI-control experiments in this
fork. To reproduce the Mars Rover results from the original paper, use the
solver from the authors' original APPROPO repository.

## APPROPO PI Control Experiment

This fork adds a PI control feasibility experiment that uses APPROPO with an
actor-critic positive-response oracle. The controller is a bounded PI policy,
parameterized by `Kp` and `Ki`, for a first-order step-tracking plant. The
measurement vector is:

```text
[reward, -response_time]
```

where the reward is the negative of tracking error plus weighted actuator
effort, and response time is the first time the output enters the tolerance
band around the setpoint.

The target set is:

```text
reward >= -13.5
response_time <= 3.0
```

Run the main PI actor-critic APPROPO experiment:

```bash
python3 -B -m ApproPO.appropo_pi_real \
  --seed 96656 \
  --num_epochs 80 \
  --rl_iter 80 \
  --rl_traj 24 \
  --check_traj 8 \
  --cache_size 12 \
  --mx_size 5 \
  --proj_lr 0.4 \
  --olo_momentum 0.5 \
  --pi_ac_gain_step 0.04 \
  --plot \
  --plot_output results/pi_actor_critic_appropo_seed96656_long.png
```

Generate the mixture/measurement-space summary plot:

```bash
python3 -B -m ApproPO.plot_pi_mixture \
  --seed 96656 \
  --num_epochs 80 \
  --rl_iter 80 \
  --rl_traj 24 \
  --check_traj 8 \
  --cache_size 12 \
  --mx_size 5 \
  --proj_lr 0.4 \
  --olo_momentum 0.5 \
  --pi_ac_gain_step 0.04 \
  --summary_output results/pi_actor_critic_mixture_seed96656_long.png
```
