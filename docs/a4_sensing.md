# A4 sensing options (branch `2026-11-16-a4-sensing`, off `2026-11-03-bottlecap` 4124840)

Student-side sensing ablations for the touch-necessity paper. All options are `env.tactile.*` overrides, off by default.
They change only what a student can read; the teacher obs, rewards, terminations and every reader of
`env._last_tactile` (bottle strip rule, retract / release checks, eval panels) are untouched.

| Override | What the student gets | Buffer / obs field |
|---|---|---|
| `student_res=4\|2\|1` | Fingertip pads block-averaged (weighted by rays per taxel) to 4x4 / 2x2 / 1 taxel, upsampled back to 8x8 (same tensor and encoder; off-pad taxels stay 0). 1 = the pad's mean contact depth. | `env._student_tactile`, field `tactile` |
| `student_hand_grid=2\|1` | Whole-hand skin at 2x2 / 1 taxel per link (block max), upsampled back to 4x4. | `env._student_tactile_hand`, field `tactile_hand` |
| `student_hold_steps=k` | Sensor rate 60/k Hz: every student touch / force buffer refreshes every k policy steps and at reset. | all of the above and below |
| `wrench=true` | Pooled touch: net force + torque per pad (5 x 6) and per skin link (21 x 6), from the same rays as the maps (f = depth x inward ray direction; F in mm of depth averaged over the pad, torque in mm x cm), pad / link frame. Depth only, no shear (like the maps). | `env._student_tactile_wrench`, `env._student_tactile_hand_wrench`; fields `tactile_wrench`, `tactile_hand_wrench` |
| `joint_ft=true` | PhysX joint reaction wrenches (N, N m): wrist at iiwa a7 (hand merged into link 7) and the 5 DIP joints (fingertip F/T incl. friction). | `env._student_wrist_ft`, `env._student_tip_ft`; fields `wrist_ft`, `tip_ft` |

- The new fields can be listed in `student_obs.proprio_list` (DAgger depth / RGB students) or `obs.obs_list` (teacher-level tests);
  `policy_override=zero|random` also applies to them.
- Offline: `utils/sensing.py` `pool_pads(taxels, 8, sub, ray_counts)` and `pool_skin(skin, 4, sub)` are the same functions
  the env uses, so they apply to recorded taxels and give exactly the env's result (the bins nest). `ray_counts` =
  `env._tactile.ray_counts` (rays per taxel; save it once per map).
- The residual ACT adaptor (`residual/online/train_dagger.py`, `bc/act/inference_act_screw.py`) still reads
  `u._last_tactile`; to use a variant it must read `getattr(u, "_student_tactile", u._last_tactile)` (a one-line change
  in A5 / A2 files, not made). Pooled / F/T fields need a low-dim token path in the adaptor (A5).
- Robot links have gravity disabled in this sim, so `wrist_ft` carries no hand weight (like a tared, gravity-compensated
  sensor): only contact and inertial / actuation loads.

Checks (`FoundationTouch/tools/sensing/smoke_sensing.py`, results in `export/agents/a4-sensing.md`).
