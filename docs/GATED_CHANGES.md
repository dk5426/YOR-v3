# Gated teleop/solver changes (2026-08-25 wave)

Every change ships OFF by default -- a run with no new flags is bit-identical
to the `pre-gates` checkpoint (verified: 40-solve trajectory, max |Δq| = 0.0).

**Restore everything:** `git reset --hard pre-gates` (also branch `backup-pre-gates`).

| Gate | Where | Flag | Off / try |
|---|---|---|---|
| T1 clutch reseed | teleop client | `--clutch-reseed` | off / on |
| T2 arm dispatch deadband | yor.py | `--arm-joint-deadband RAD` | 0.05 / 0.005 |
| S1 null-space home attractor | yor.py | `--nullspace-home-gain G` (+ `-weight`, `-max-vel`) | 0 / 0.3 |
| S2 target leash | yor.py | `--target-leash-m M`, `--target-leash-rad R` | 0 / 0.15, 0.8 |
| S3 constrained primary | yor.py | `--constrained-primary` | off / on |
| S4a task weighting in dls | yor.py | `--dls-task-weighting` | off / on |
| S4b adaptive damping | yor.py | `--dls-adaptive-damping SIGMA` (+ `--dls-damping-max`) | 0 / 0.05 |
| S5a swivel parallel reference | yor.py | `--swivel-parallel-ref` | off / on |
| S5b swivel re-latch | yor.py | `--swivel-relatch-err RAD` (+ `--swivel-relatch-time`) | 0 / 1.57 |
| S5c relatch RPC | both nodes | `relatch_elbow_swivel(side=None)` | always available |
| S6 joint7 posture A/B | yor.py | `--no-posture-stiffen-joint7` (pre-existing) | on / off |
| S7 solver diagnostics | yor.py | `--no-solver-diagnostics` | ON by default |

yor.py prints `[yor] experiment gates: ...` at startup so every console log
records which gates a session ran; the trajectory CSV headers carry the same.

New CSV columns (S7): `l_sigma_min r_sigma_min l_manip r_manip l_swivel
l_swivel_tgt l_swivel_err r_swivel r_swivel_tgt r_swivel_err collision_rows`.
Reading a stuck event: sigma_min ~ 0 -> singularity; manip low -> reach limit;
|swivel_err| large & steady -> fought elbow branch; collision_rows up -> the
solver is being constrained, not confused.

Interplays to respect while A/B-ing (one change at a time, as always):
- S1 creep must clear the dispatch deadband -> pair S1 with T2.
- S4a rescales J against lambda -> re-judge S4b thresholds if both are on.
- S5b is the backstop for S5a's slow reference drift -> enable together.
- S2 leash also bounds the solver's per-tick error race -> reduces the events
  S5b exists for; this is expected, not a masking bug.

Suggested progression (each step floor-tested before the next):
1. `--arm-joint-deadband 0.005` alone (T2)
2. `+ --clutch-reseed` on the client (T1)
3. `+ --target-leash-m 0.15 --target-leash-rad 0.8` (S2)
4. `+ --nullspace-home-gain 0.3` (S1)
5. `+ --constrained-primary` (S3) -- judge backward/inward motion specifically
6. `+ --swivel-parallel-ref --swivel-relatch-err 1.57` (S5)
7. `+ --dls-task-weighting`, then `--dls-adaptive-damping 0.05` (S4) -- judge rotation
