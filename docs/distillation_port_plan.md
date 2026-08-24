# Port plan: DEXTRAH DAgger distillation → play2perfect / `tight_insertion`

**Goal.** Distill play2perfect's state-based `tight_insertion` RL teacher into a
depth-vision behaviour-cloning student, reusing DEXTRAH's DAgger machinery
(`~/DEXTRAH`, NVlabs, clean at `ebc08ed`).

**Status.** Environment set up and validated; task selected; one latent bug in the
student camera path found and fixed. **Phases 1 and 2 are done and verified**
(`check_phase1_obs.py` 23/23; `check_phase2_teacher.py` gate PASSED at 96.9%;
`check_phase2_student_env.py` all checks pass). Phases 3–6 not started.

---

## 0. Established facts

### Task selection — `tight_insertion`

Teacher success rates, measured with `evaluation/eval_offline.py` at 1024 envs
(scores each env's first episode only, discards episodes under 100 steps as
unstable inits):

| problem | trials | bad init | success | goal ratio | retract |
|---|---|---|---|---|---|
| `beam_assembly_step1` | 976 | 48 | 97.1% | 97.6% | 97.1% |
| **`tight_insertion`** | 911 | 113 | **96.9%** | 97.1% | 96.9% |
| `beam_assembly_step2` | 890 | 134 | 92.2% | 92.4% | 92.2% |
| `screwing` | 1022 | 1 | **61.4%** | 75.1% | 53.9% |

`screwing` is ruled out: a student cannot exceed its teacher, and 61.4% on the
longest-horizon task (10 subgoals, 720° of coupled screw motion, 10.8 mm hole,
87.5 mm object) is not a usable teacher. Note its `bad_init` is 1 — those are
genuine policy failures, not unlucky resets.

`tight_insertion` and `beam_step1` are tied within noise at n≈900–1000, so teacher
quality does not break the tie. The tiebreakers do:

| | `tight_insertion` | `beam_step1` |
|---|---|---|
| subgoals | 2 | 2 |
| object geometry | **0 meshes** (4 box primitives) | 36 mesh hulls + SDF |
| receptacle geometry | **0 meshes** (6 boxes) | 2 meshes + SDF |
| render/sim cost | **lowest of the four** | higher |
| dormant student raycast prim path | **already matches** `/Hole/hole/visuals` | needs `/Hole/plate/visuals` |
| object size | 250 mm (easiest to see) | 229 mm |

### Precision budget is ~2 mm, not 0.5 mm

The teacher trained with `goal_xy_obs_noise: 0.002` subtracted from its
`keypoints_rel_goal` observation, so it is explicitly robust to 2 mm of
goal-estimate error; its learned compliance closes the remaining 0.5 mm
mechanical clearance. **2 mm hole localization is the student's spec.**

### Throughput (measured, RTX 5090 32 GB)

`tight_insertion`, mono depth 160×90, TiledCamera, depth aug on:

| envs | camera | step Hz | step+`get_student_obs()` Hz | env-steps/s |
|---|---|---|---|---|
| 256 | **off** | **95.7** | 95.7 | 24,490 |
| 128 | on | 32.6 | 23.5 | 3,011 |
| **256** | **on** | **23.1** | **17.4** | **4,450** |
| 512 | on | 15.1 | 10.8 | 5,532 |
| 1024 | on | — | **OOM: 56 GB RSS, killed by kernel** | — |

- **Camera tax is 4.1×** (95.7 → 23.1 Hz at 256 envs). `get_student_obs()`
  (preprocess + augmentation + delay queues) costs a further 1.33× → **5.5× total**.
- Larger batches are more sample-efficient (env-steps/s rises with env count),
  but the *iteration* rate falls.
- DAgger takes **one gradient step per iteration**, so iteration rate = gradient
  steps/s. For DEXTRAH's default 100k iterations: **256 envs ≈ 1.6 h**,
  512 envs ≈ 2.6 h.
- **1024 envs does not fit on this machine** (56 GB resident, global OOM).

**Decision: start at 256 envs.** Matches DEXTRAH's setup so their constant 1e-4 LR
is in the right regime, and 17.4 Hz keeps the debug cycle tolerable. Move to 512
after it trains if gradients look noisy — 2× batch, +24% sample throughput,
+1 hour. Note `num_envs` **is** the batch size here; it is not a free knob.

### Bug already fixed

`StudentObsCfg.camera_quat_wxyz` defaulted to identity, which under
`camera_convention="ros"` (+Z = optical axis) aimed the camera **straight up at
the sky**. Every pixel fell past `depth_max_m` and window-normalized to a constant
`1.0` — verified `n_unique == 1` across all envs. Replaced in
`isaacsimenvs/tasks/play/play_env_cfg.py` with a validated look-at pose
(`camera_pos=(0.0,-0.75,0.95)`, `camera_quat_wxyz=(-0.505666,0.862729,0.0,0.0)`,
aimed at the table top `(0,0,0.53)`), derived via
`create_rotation_matrix_from_view` (returns OpenGL) +
`convert_camera_frame_orientation_convention(opengl→ros)`.

This is hard proof the student camera path **had never once been executed**.
Treat everything under `student_obs` as untested, not working.

---

## 1. The structural finding that shapes the port

**DEXTRAH's `Dagger` takes the raw gym env, not the rl_games vec wrapper.**

```python
env = gym.make(args_cli.task, cfg=env_cfg)   # run_distillation.py:105
ov_env = env.env
dagger = Dagger(env, dagger_config, ...)     # reads _get_observations() verbatim
```

rl_games is used only for *model building* (`model_builder.register_network`,
`ModelBuilder().load(params).build(cfg)`) — never for env wrapping.

Consequences:
- We do **not** need `_DAggerRlGamesVecEnvWrapper` (`rlgames_utils.py:273`). It
  exists in play2perfect but is irrelevant to this loop. Leave it alone.
- `_get_observations()` may return any keys — no `clip_obs` routing, no
  `obs_groups` restriction.
- Therefore **emit `img` and `proprio` as separate keys** rather than the single
  flattened vector play2perfect builds today. The conv encoder wants NCHW;
  flatten→reshape is pointless indirection.

### Obs-dict contract

| key | DEXTRAH | play2perfect today | port target |
|---|---|---|---|
| student proprio | `policy` (159) | `policy` = image_flat ++ proprio (14487) | `proprio` (87) |
| student image | `img` (N,1,H,W) | *(inside the flat vector)* | `img` (N,1,90,160) |
| teacher obs | `expert_policy` (167+N) | `teacher_obs` (140) | `teacher_obs` (140) |
| critic | `critic` (214+N) | `critic` (162) | `critic` (162) |
| aux targets | `aux_info` = {`object_pos`} | **absent** | `aux_info` (below) |
| seg mask | `mask` | absent | skip (RGB-aug only) |

### Aux targets — do not copy DEXTRAH's choice

DEXTRAH regresses `object_pos` because its object sits free on a table and its
pose is the unknown. **In `tight_insertion` the object is already grasped**, so its
pose is largely recoverable from proprio; the genuinely hidden quantity is the
**hole pose**, randomized per episode over ±187.5 × ±100 mm (`hole_x_range`,
`hole_y_range`; yaw fixed at 0 by default).

Recommended `aux_info`:
- `hole_pos` (3) — the primary unknown. Available as `env.hole_pos`.
- `keypoints_rel_goal` (12) — the teacher's literal decision input; the most
  task-aligned supervision available.
- `object_pos` (3) — cheap, helps ground the encoder.

Aux labels are sim ground truth, so they are **immune to DAgger covariate shift** —
correct however off-distribution the student wanders. With a high `aux_coeff`
(DEXTRAH uses 10) early training is dominantly a supervised vision problem.

---

## 2. Phases

### Phase 1 — env: emit what the student needs — **DONE**

Landed:
- `precise_assembly_env.py::_get_observations` returns `proprio` (N,87) / `img`
  (N,1,90,160) / `teacher_obs` (N,140) / `critic` (N,162) / `aux_info` as
  separate keys; no more flatten→slice→reshape round trip.
- `PreciseAssemblyEnv._aux_info()` → `hole_pos` (3), `keypoints_rel_goal` (12),
  `object_pos` (3), all env-relative.
- `build_observations(env, aux_out=None)` now fills `aux_out` in place with the
  clean `object_pos` / `keypoints_rel_goal` it already computes for the critic,
  so aux labels cannot drift from the teacher's own keypoint geometry. Default
  `None` keeps the teacher path byte-identical.
- `student_obs:` block added to `cfg/task/PreciseAssembly.yaml` (28 keys, all
  validated against `StudentObsCfg`), `enabled: false`.
- Hard guard: `student_obs.enabled` + `random_goal_fraction > 0` now raises,
  before `super().__init__()`. Random-goal envs set `hole_pos` to a
  `(0, 0, -1)` sentinel, which would silently poison the primary aux target.
- `tools/distillation/check_phase1_obs.py` — acceptance check. Includes a
  permanent `img.unique() > 1` assertion, the exact test that would have caught
  the sky-camera bug.

**Deviation from the original plan:** the `observation_space` Dict override was
*kept* (and corrected to the new keys) rather than deleted, so the declared
space matches what the env actually emits. `aux_info` is deliberately absent
from it — supervision, not observation. The stale `cfg.observation_space =
student_dim` mutation *was* removed (it would have sized `reset_utils`'
`_obs_queue` to 14487 on any post-init reallocation).

> **Correction (found in Phase 2).** The original justification given for
> keeping the Dict — that `register_rlgames_env` needs the `teacher_obs` key to
> select `_DAggerRlGamesVecEnvWrapper`, and hence to give `teacher_env_info()` a
> bounded `teacher_obs_space` — was wrong. `RlGamesVecEnvWrapper.__init__`
> *raises* unless the env exposes a `"policy"` key
> (`isaaclab_rl/rl_games.py:128-130`), and the student contract emits
> `proprio`/`img` instead. So that wrapper cannot be constructed at all when
> `student_obs` is enabled; it would have failed loudly, not silently. Phase 2
> builds `env_info` directly from dims instead, which is simpler anyway. Keeping
> an accurate space is still right, just for the plain reason.

Measured en route: the teacher's noisy `keypoints_rel_goal` differs from the
clean aux label by max 0.00188 — independent confirmation of the 2 mm
`goal_xy_obs_noise` precision budget above.

### Phase 2 — teacher wrapper — **DONE, GATE PASSED**

DEXTRAH builds its teacher with a bare `network.build()`. **This will fail here.**
`PreciseAssemblySAPG.yaml` sets `fixed_sigma: coef_cond`, which makes sigma a
lookup table requiring `coef_ids` + `coef_id_idx` at build time
(`network_builder.py:289-292`) and reads a block-id column appended to the obs at
forward time (`network_builder.py:410`). DEXTRAH supplies neither.

Use rl_games' `PpoPlayerContinuous`, which wires all of it automatically
(`players.py:54-55`: `coef_ids = ids[::num_agents,0]`,
`coef_id_idx = obs_shape[0]`), plus `running_mean_std` restoration and LSTM state.
`eval_offline.py` already proves this path loads the released checkpoints.

```python
runner = Runner(); runner.load(agent_cfg); runner.reset()
player = runner.create_player()
player.set_weights(_load_checkpoint_weights(player, ckpt))
```

Feed it via the existing `teacher_env_info(wrapped)` helper
(`rlgames_utils.py:309`). DAgger needs `mus`/`sigmas`, not `get_action()`'s
rescaled output — so call `player.model(input_dict)` directly and read
`res_dict["mus"]`. **That pattern is already written** in
`deployment/rl_player.py:141-174`, including the block-id append
(`torch.cat([obs, 50.0*ones])`, `rl_player.py:99`). Reuse it.

Thread `player.states` (LSTM) across steps; zero on episode boundaries.

**Gate: teacher `mus` must reproduce 96.9%.** If they are wrong, every downstream
loss is meaningless but will present as a convergence problem.

---

#### Phase 2 outcome

Landed as `isaacsimenvs/distillation/teacher.py` (`Teacher`,
`teacher_env_info_from_dims`), gated by
`tools/distillation/check_phase2_teacher.py` and
`tools/distillation/check_phase2_student_env.py`.

**Gate result — tight_insertion, 1024 envs, paired initial states:**

| block_id | trials | bad_init | success | goal ratio |
|---|---|---|---|---|
| **50.0** (constant) | 881 | 143 | **96.9%** | 97.2% |
| 0.0 (constant) | 889 | 135 | 95.4% | 95.7% |
| `ramp` (player-equivalent) | 878 | 146 | 96.6% | 96.9% |

96.935% vs eval_offline's 96.9% — the `mus` path reproduces the teacher to
within 0.04 pp. **Gate passed.**

**No rl_games env wrapper.** `teacher_env_info(wrapped)` is not usable here (see
the Phase 1 correction above): `RlGamesVecEnvWrapper` requires a `"policy"` obs
key. `teacher_env_info_from_dims()` builds `env_info` from dims + the agent
yaml's `clip_observations`/`clip_actions` instead. The action space must be
finite or `rescale_actions` NaNs every action.

**The block-id column is a per-env ramp, not a constant.** `BasePlayer` builds
it as `linspace(50, 0, num_envs)` (`player.py:93`) and appends it in
`env_reset`/`env_step` (`player.py:208, 258`) — so under eval_offline env *i*
literally gets a different teacher than env *j*. That is wrong for
distillation, since the student cannot observe the block id, making an
env-varying teacher irreducible label noise. The constant 50.0 that
`deployment/rl_player.py:99` uses turns out to be *better* than the ramp, not a
compromise. Note the wrapper clips the obs and the block id is appended
afterwards, so 50.0 must survive a `clip_observations` of 10.0 — do not clip
after appending.

**`sigmas` are obs-independent, and the selected block's are unusable as loss
weights.** `sigma = sigma_act(sigma[idxs])` (`network_builder.py:411`) indexes
only on the block id, so for a fixed `block_id` the returned sigma is a
*constant 29-vector* — it carries no state-dependent teacher uncertainty at all.
(Confirmed empirically: 4 envs with different observations returned identical
sigmas.) The per-block table, with the `1/sigma_T^2` weight DEXTRAH's loss
implies:

| row | coef_id | sigma_min | sigma_max | sigma_med | w_min | w_max |
|---|---|---|---|---|---|---|
| 0 | 50.0 | 0.162 | **170.4** | 1.809 | 3.4e-05 | 3.8e+01 |
| 1 | 40.0 | 0.151 | 33.16 | 1.609 | 9.1e-04 | 4.4e+01 |
| 2 | 30.0 | 0.135 | 3.374 | 1.332 | 8.8e-02 | 5.5e+01 |
| 3 | 20.0 | 0.120 | 1.380 | 0.926 | 5.3e-01 | 6.9e+01 |
| 4 | 10.0 | 0.114 | 1.194 | 0.801 | 7.0e-01 | 7.7e+01 |
| 5 | 0.0 | 0.115 | 1.222 | 0.819 | 6.7e-01 | 7.6e+01 |

coef_id is monotone in exploration magnitude — 50.0 is the high-exploration
block. Its sigma spans **1054x**, i.e. **1.1e6x** in `1/sigma^2`, so DEXTRAH's
weighting would silently zero the loss on whichever joints that block is noisy
on. **Phase 5 decision: take `mus` from block 50.0 (best success rate) but do
NOT weight by its `sigmas`.** Use uniform weights, or borrow a low-exploration
row (4 or 5, both well-conditioned) if per-joint weighting is wanted. The
`+ L2 on sigmas` term in DEXTRAH's loss is also near-pointless here — it
regresses the student's sigma toward a constant.

### Phase 3 — viser image panel

`_viser_demo.py` streams a 17-tuple of poses and never shows camera frames. Add
`server.gui.add_image` fed from `get_student_obs()["image"]`. Small change,
highest debugging value in the project — you cannot diagnose a vision policy
without seeing its input, especially with depth augmentation on.

### Phase 4 — student network

Port `dextrah_lab/distillation/a2c_with_aux_cnn.py` (the **mono depth** variant,
matching play2perfect's `image_modality="depth"` default) to
`isaacsimenvs/distillation/a2c_aux_cnn.py`.

- `Network.forward(obs_dict)` reads `obs_dict['obs']` (proprio) + `obs_dict['img']`.
- `CustomCNN` (`a2c_with_aux_cnn.py:215`) is a 4-layer conv stack — fine for 160×90.
- Aux heads: shared `aux_mlp`, then one `nn.Linear` per `aux_outputs` entry
  (`a2c_with_aux_cnn.py:338-357`).
- Register via `model_builder.register_network("a2c_aux_cnn_net", Builder)`.

New `isaacsimenvs/cfg/train/PreciseAssemblyStudent.yaml`, modelled on
`rl_games_ppo_lstm_scratch_cnn_aux.yaml`, with
`aux_outputs: {hole_pos: {size: 3}, keypoints_rel_goal: {size: 12}}`.

Skip for v1: `stereo_encoder.py`, all `a2c_*_transformer*.py`, `rgb_augs.py`
(play2perfect already has a depth-aug pipeline; RGB needs texture assets).

### Phase 5 — DAgger loop

Port `distillation.py` → `isaacsimenvs/distillation/dagger.py`. Keep the loop
shape, with four deliberate deviations:

1. **Drop DDP / `torch.distributed`** for v1. DEXTRAH hard-requires `WORLD_SIZE`,
   `RANK`, `LOCAL_RANK` (`distillation.py:101-103`) and wraps in DDP. Single-GPU
   first.
2. **Do not hardcode `beta = 0.`** `distillation.py:376` clobbers a fully-written
   15k-iteration teacher-driven warmup. DEXTRAH gets away with β=0 because its
   *geometric fabric* bounds the reachable state set — 11 bounded actions through
   a damped second-order controller means a garbage policy still produces smooth,
   collision-free motion. **play2perfect has no fabric**: 29 raw joint targets
   mean a cold student can reach genuinely unrecoverable states, and the teacher's
   LSTM hidden state goes off-distribution along the student's trajectory.
   **This is the single most important deviation.**
3. **Do not hardcode `seq_length = 1`.** `distillation.py:177` reads the config's
   value (20) then discards it, giving BPTT length 1 — the recurrent weights never
   learn to write a predictively-useful hidden state. Start from the config; tune
   down only if memory forces it.
4. Drop the per-step `torch.cuda.empty_cache()` (`distillation.py:540`) and the
   dead done-time flush (`:572`, unreachable when `seq_length == 1`).

Loss: weighted L2 on teacher `mus` (weights `1/sigma_T²`) + L2 on `sigmas` +
`aux_coeff · Σ aux`. Reduction is `mean` over the env batch
(`torch_ext.apply_masks` with `mask=None`).

### Phase 6 — entry point

`isaacsimenvs/distill.py`, modelled on `train.py`: AppLauncher first (must pass
`--enable_cameras`), then `gym.make`, register networks, build teacher, run
`Dagger.distill()`.

---

## 3. Risks and open questions

1. **`get_student_obs()` is newly-exercised code.** The sky-pointing camera proves
   it had never run. Expect more latent bugs in the delay queues
   (`_apply_student_tensor_delay`, `_apply_student_bundle_delay`), the crop path,
   and the stereo/raycaster backends.
2. **11% bad inits.** `tight_insertion` discards 113/1024 first episodes as
   unstable (object falls / ungraspable at reset). `eval_offline` drops them;
   DAgger *will* visit them and teacher labels there are near-worthless. Consider
   `enable_dropped_on_table_term` (off by default) so they end fast.
3. **Framing is loose.** ~35% of pixels land on the scene; the hole is ~10×10 px
   at 160×90. Comparable to DEXTRAH, so workable — do not tune prematurely. If
   localization stalls, move the camera rather than changing
   `focal_length`/`horizontal_aperture`, which are documented as matched to a real
   ZED HD1080 calibration (lab serial 15107).
4. **Memory ceiling.** Cap any sweep at 512 envs; 1024 OOMs the machine.

## 4. Order of work

~~Phase 1~~ → ~~Phase 2 (gate passed)~~ → **Phase 3** → Phase 4 → Phase 5 → Phase 6.

Phases 1 and 2 are independent and can be done in either order; Phase 2 is the one
that could invalidate assumptions, so it is worth front-loading.
