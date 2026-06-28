# Sim-to-real deployment

This release ships a **minimal** deployment reference — the reusable policy-inference
core, not a full hardware stack. The same object-centric policy trained in Isaac Sim runs
on hardware unchanged: it consumes a state observation and emits normalized joint targets.

## What's included

- `deployment/rl_player.py` — `RlPlayer`, a thin wrapper around the vendored `rl_games`
  player. It loads a training `config.yaml` + `model.pth`, manages the policy's recurrent
  (LSTM) state, and maps observations → normalized actions.
- `deployment/rl_player_utils.py` — config loading helpers.

## `RlPlayer` at a glance

```python
from deployment.rl_player import RlPlayer

player = RlPlayer(
    num_observations=OBS_DIM,      # must match the trained policy
    num_actions=NUM_ACTIONS,
    config_path="pretrained_assembly/<problem>/config.yaml",
    checkpoint_path="pretrained_assembly/<problem>/model.pth",
    device="cuda",
    num_envs=1,
)

player.reset()                              # clears the LSTM hidden state (call per episode)
action = player.get_normalized_action(obs)  # obs: (num_envs, OBS_DIM) -> action in [-1, 1]
```

The observation vector must be assembled with the **same layout** the policy was trained on
(see the observation construction in `isaacsimenvs/tasks/play/utils/obs_utils.py` and the
precise-assembly env). Normalized actions are rescaled to joint targets by your robot driver
exactly as the sim action pipeline does (`isaacsimenvs/tasks/play/utils/action_utils.py`).

## Wiring it to a robot

A typical real-world loop mirrors the sim control loop:

1. **Perceive** — estimate the manipulated object's 6-DoF pose and read the robot's joint
   state. For the pose tracker we use SAM + FoundationPose; see our
   [FoundationPose fork](https://github.com/kushal2000/FoundationPose) for setup and usage
   (install it in a separate environment).
2. **Build the observation** — same fields and ordering as training.
3. **Act** — `player.get_normalized_action(obs)`; rescale to joint targets.
4. **Command** — send targets to the arm + hand controller at the control rate; repeat.

This release does not include the authors' specific ROS nodes, camera drivers, or pose
trackers, which are hardware-dependent. `RlPlayer` is the piece you reuse; the perception
and robot I/O around it are yours to provide.
