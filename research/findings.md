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
