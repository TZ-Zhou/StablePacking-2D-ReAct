# Research findings: Observation V3 signals (P0–P2), 2026-11-08

All offline, on the 50 walls of `thangavelu-5arms-10seeds` (formal quality: `wang-quality-v1`) and
the 6 counterfactual branch points (`paper-hardstone-01`, `smoke-001-cf`). Exploratory: few walls,
few branch points. Decision: these directions are paused; none is wired into a controller or
into Observation V2.

## Quality method sensitivity (wang-quality-s1)
- 02 leaves out 13 overlapping stones (sim-grid neighbours that touch; 4–12 px overlap bands at
  the original resolution). Drawing them (s1: contested pixels to the later stone, s1-earlier: to
  the earlier) changes only F_SF of those 13 walls (0.36–0.41 → 0.40–0.43).
- Arm ranks unchanged for all four metrics; paired counts move by at most one seed. v1 stays formal.
- Low F_SF in v1: 13 walls from 02's leave-out, 4 walls from unplaced stones (5 stones).

## P0: single-step preview (placement_preview.py, placement_preview_study.py)
- Partial-wall F_AV correlates with final F_AV late in the wall (rho 0.61 / 0.71 after 12 / 15 of
  18 stones), mostly because the wall is mostly built. Density does not predict F_SF; nothing local
  predicts F_LR. Band variants are mostly undefined.
- At decision level (6 branch points) preview differences agree with final differences about as
  often as not (e.g. av → F_AV 8 / 10).
- Greedy-Preview (Pareto on density, ah, av, LLC vs Paper) leaves Paper at 23% of 754 decision
  points, with no evidence that this helps.

## P1: limited lookahead (lookahead_study.py; 17 runs in research/runs/lookahead-p1, untracked)
- Each lookahead reproduced the existing wall's trace event for event (RNG, stones, settings).
- 3 of 6 points are within 2 placements of the wall's end: there the lookahead is a full rollout.
- On the 3 genuinely partial points (14 pairs) k = 2 / 4 is not consistently better than k = 0:
  sf → F_SF 11/3 at k = 2 but 3/11 at k = 4; F_LR about even; completion never differs within
  4 placements. Cost: about 60 s (k = 2) / 91 s (k = 4) per alternative, about 9× / 14× a wall.

## P2: Pareto and hard stones (decision_signals.py)
- Pareto front of the five weighted filter scores keeps about 43% of feasible candidates and
  every outcome class of the 120 orders (as it must: a cascade pick is never strictly
  dominated). It excludes no known-better alternative, and it cannot discriminate between orders.
- Hard-stone ranking (past infeasible samplings, last pose count, area): the 5 unplaced stones
  rank 1, 6, 1, 3, 4 of 6 when 6 remain. At the branch points the stone whose placement improved
  completion ranks easy at decision time (it had poses then), so the signal would not pick it.
- Cost: Pareto 0.17 ms, hardness 0.02 ms per event.

## Run time of one case (research/perf)
- `fft-d2-0` (100 stones, 41 placed, `paper` order 3,0,2,1,4, seed 20), one run on a 16-logical-core
  machine: with the per-batch Pool of `build_wall` 406 s, of which 359 s is the build loop; about 98% of
  that loop is Pool creation and worker start-up (52 batches, about 6 s each), while the workers compute
  about 48 s of CPU in total. With one persistent Pool 55 s.
- Worker CPU is 88% `evaluate_kine`; MOSEK's `optimize` is about 14% of worker CPU, the rest is Python
  model assembly (`cal_Aglobal`, `clear_contps`, the loops in `solve_force_rigid`). Not touched.
- After the persistent Pool the largest items are stone preparation (`rotate_min_shape_factor`, 21 s of
  55 s), the build loop (24 s) and the figure output (`draw_assembly`, final mechanism plot, about 9 s).
- Equivalence: one persistent run against one per-batch run of the same case, policy, order and seed, and
  four simultaneous persistent runs against that persistent run, gave identical `decisions.txt`,
  `filter.txt`, `transformation.txt` and `base.pkl`. Not verified: other cases (`bricks`,
  `thangavelu-noise80`, `almeida-pr` with `two_sides` and `sample_twice`), other policies and seeds
  beyond one comparison of two seeds, and persistent-worker state across calls (not audited).
  A broader check was started and stopped unfinished.

## Temporal myopia on TH_T80 (recorded walls)

Does the step-by-step cascade miss the long-term result? All offline from `research/runs` traces; final metrics from
`wall_metrics.wall_quality` (`corrected`). Exploratory: one scene, 18 stones, tests uncorrected. Scripts:
`research/analysis/myopia_tables.py`, `myopia_report.py`, `surface_fit_check.py`; whether they reproduce every number
below is not yet checked (`docs/TODO.md`). What follows from it: `docs/plans/surface-fit-heuristic.md`.

### Sample
- 101 finished walls; 72 remain after deduplicating on seed plus placement sequence (null = Paper, lookahead "recorded",
  determinism repeats); 13 seeds; 62 of 72 place all 18 stones. Pooling the duplicates inflates LLC → F_LR concordance
  from 22/34 to 84%.
- 72 walls: F_LR 0.087–0.870 (mean 0.311, SD 0.161); height 161–258 px (mean 199.6, SD 23.75). Spearman F_LR–height
  0.02, F_AV–height −0.71.

### Filter order (50 walls of `thangavelu-5arms-10seeds`)
- Share of variance, arm / seed / residual: F_SF .04/.25/.71, F_AH .08/.24/.68, F_AV .17/.17/.66, F_LR .06/.41/.53,
  height .09/.28/.63, n_placed .12/.14/.74.
- No arm beats Paper on F_LR. `code_original` vs Paper: F_AV +4.38 (8/10 seeds, p=.04), height −16.4 px (7/10, p=.23).
- 17 of 50 walls are Pareto-optimal on F_SF, F_AH, F_AV, F_LR.

### Immediate scores vs final outcome (74 unique branch points, 195 wall pairs, 12 seeds; not causal)
The branch with the higher immediate score ends better (F_LR; height; n_placed):
- LLC 22/34; 16/34; 4/10. NHR 34/73; 42/73; 8/26. IWR 22/49; 24/49; 9/18. GDS 30/57; 28/57; 11/19.
- LFR 34/74; 37/74; 21/27 (p=.006). Sum of the five 36/74; 35/74; 7/27 (p=.019). Thirty tests in all, F_AV and F_AH included.
- At the 40 LLC-tied points the other scores agree 35–63% of the time, with no stable direction.
- 51 points lie at t ≤ 6 and only 5 at t ≥ 13.

### Action reach
- Feasible candidates per step: median 14 (mean 13.4). The 120 orders reach a median of 3 (mean 3.24); one outcome only on
  15.8% of steps; the chosen candidate has the step's highest LLC on 71.8%.
- A feasible stone is picked by at least one order (t 1–6 / 7–12 / 13–18): five largest 22.5 / 26.9 / 59.7%, other
  stones 40.4 / 55.6 / 75.2%, stone 13 22.9 / 21.2 / 52.2%. 45.4% overall.

### Stone consumption
- Area vs mean placement step: Spearman 0.61 (p=.007); the five largest stones are on average the last five; 83% of walls
  place large stones later.
- Within seed, the later the five largest go, the taller the wall (+0.24, p=.046), the worse F_AH (+0.30, p=.009) and the
  more stones placed (+0.25, p=.031). No causal claim.

### Endgame
- Final height exceeds the ideal by 89.0 px on average: 64.7 from the uneven top, 24.3 from void under the skyline.
- The final peak is set by the last placement on 45 of 72 walls and by the second-last on 19.
- At t = 13 (62 complete walls): peak 114.6, mean height 75.0; final height 200.3, so the last six stones add 85.7 px
  (SD 26.5). Spearman with final height: peak at t = 13 0.37, later gain 0.65.
- Feasible poses at t = 13–18: 12.7, 9.8, 8.0, 5.6, 3.6, 1.9 (16–19 mid-wall). Chosen LLC falls from 0.98 (t = 2) to 0.33
  (t = 17–18).
- Ten incomplete walls leave 12 stones unplaced: stone 13 six times, stone 8 three, stone 16 two, stone 9 one.

### Size of one divergence
- Median |final difference| by divergence step (1–3 / 4–6 / 7–12 / 13–18): F_LR .10 / .14 / .12 / .17; height 14.9 / 13.5 /
  22 / 15 px. Includes later policy differences.
- 11 non-neutral causal branches at the 6 hand-picked points: F_LR better 8, n_placed +1 in 7, height worse 8 (1 tie),
  F_AV worse 7, F_AH worse 8.

### Surface features at step T vs final result (within-seed Spearman, n = 72)
T = 3 / 6 / 9 / 12 / 15:
- Peak minus mean height: height −.17 / +.12 / .00 / +.22 / +.26; F_AV −.11 / −.34 / −.35 / −.28 / −.36; F_AH −.03 / +.05 / +.17 /
  +.36 / +.38; F_LR −.16 / −.07 / −.11 / −.08 / −.04.
- Cumulative GDS: height +.08 / −.18 / −.25 / −.40 / −.42; F_AV +.22 / +.35 / +.37 / +.34 / +.35.
- Cumulative LLC on F_LR: +.20 / +.26 / +.27 / +.32 / +.38. Cumulative IWR on F_LR: −.21 / +.05 / −.31 / −.30 / −.34.

### Search cost and restart baseline
- Pose evaluations to enumerate the last k stones along the recorded path, median (p90): k = 3 142 (228), 4 1,168 (2,284),
  5 10,900 (27,542), 6 148,266 (402,603). Timed part of one pose: 0.097 s CPU, convolution not timed.
- Best of N complete walls (bootstrap over 62 walls, pick by z-sum of F_LR and height), F_LR / height: N = 1 .328 / 200.6,
  N = 5 .474 / 182.6, N = 10 .551 / 180.8. Pools seeds, so it may be optimistic.

### Vertical-drop model of the skyline (1,198 recorded placements)
- The drop height equals the real height on 82.6%; 17.4% land lower (5th percentile −50 px, minimum −174), under overhangs.
  So void under the skyline is not all sealed.
- 94.6% of real placements have their centre of mass between the outermost contact columns (2 px tolerance).
- Void under a placed stone over its area: median 0.17, p90 0.585.
- Cost per candidate in ms, one process, one wall, 17 / 11 / 5 stones left: skyline envelope 0.11 / 0.24 / 0.12;
  fit of every remaining stone 11.8 / 12.9 / 4.3; largest-first drop rollout 11.7 / 14.3 / 4.1; best-fit rollout 213 / 75 / 12;
  2-D convolution of all poses 338 / 449 / 125. Predictive value of these heuristics is not tested.
