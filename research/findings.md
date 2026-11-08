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

## P0: single-step preview (preview.py, preview_study.py)
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
