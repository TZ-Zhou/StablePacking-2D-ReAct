# Research findings

Observation V3 signals (P0–P2), 2026-11-08. All offline, on the 50 walls of `thangavelu-5arms-10seeds` (formal quality: `wang-quality-v1`) and
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

### Seconds per stone after the speedup (`pref/speed-up`, 2026-11-09)
- Machine pc2, 8 workers, persistent pool, trace on, 16 full walls (8 scenes, `code_original` and `paper`), 752 stones after
  the first. Mean 2.96 → 0.35 s; stones over 2 s 386 → 0. Identical walls to the baseline `59ae057`.
- Slowest wall, `AL_IR`: mean 10.08 → 1.04 s (`code_original`), 7.25 → 0.91 s (`paper`), max 15.92 → 1.44 s. Per scene:
  `docs/plans/case-speedup-2s-per-stone.md`.
- One timing run of each wall, on a laptop; the maximum of `AL_IR` moved by 0.7 s between identical runs. An independent
  rebuild of the 16 walls had 0 stones over 2 s, slowest 1.81 s.
- `random` controller, 5 walls, identical to the baseline walls: 162 stones, 0 over 2 s, slowest 1.91 s (`AL_IR`, mean 1.25 s).

## Temporal myopia on TH_T80 (recorded walls)

Does the step-by-step cascade miss the long-term result? All offline from `research/runs` traces; final metrics from
`wall_metrics.wall_quality` (`corrected`). Exploratory: one scene, 18 stones, tests uncorrected. Reproduced by
`research/analysis/myopia_tables.py`, `myopia_report.py`, `surface_fit_check.py`. What follows from it:
`docs/plans/surface-fit-heuristic.md`.

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
- Cost per candidate in ms, one process, one wall, 17 / 11 / 5 stones left, one run on a busy machine: skyline envelope
  0.12 / 0.11 / 0.11; fit of every remaining stone 11.2 / 7.4 / 3.4; largest-first drop rollout 11.9 / 7.3 / 3.6; best-fit
  rollout 105 / 43 / 12; 2-D convolution of all poses 277 / 202 / 103. Predictive value of these heuristics is not tested.

## Factorial: filter order (3,000 walls)
5 seeds (313, 484, 717, 720, 761) × 5 scenes (`TH_T20`, `TH_T80`, `AL_R`, `AL_PR`, `AL_IR`) × 120 orders, built at
`factorial-v1` (87f8843), all `PROCESS_PASS`. Tests and their rules: `docs/plans/preregistration.md`; per-block tables in
`runs/batch/factorial-*.csv`. z is per scene, signed so that positive is better. B2 and B4 are not run.

### B3: Paper (3,0,2,1,4) against the 120 orders (descriptive, 25 blocks)
| Metric | Paper − median (median over blocks, SD) | Blocks where Paper beats the median | Best order − median |
| --- | --- | --- | --- |
| `F_SF` | −0.01 | 11/25 | 1.44 |
| `F_AH` | −0.35 | 8/25 | 1.71 |
| `F_AV` | −0.56 | 7/25 | 2.61 |
| `F_LR` | +0.72 | 16/25 | 2.50 |
- The best of 120 normal draws is expected 2.57 SD above the median, so the best order is what luck gives; the same
  best order repeats across seeds in 2 of 25 scene × metric cells.

### C2: does the score of a step predict the final wall (1,943 branch points, Holm over 24)
| Score → metric | Concordance | p (Holm) | Blocks positive |
| --- | --- | --- | --- |
| `c_LLC` → `F_LR` | 0.593 | 0.014 | 19/25 |
| `c_NHR` → `F_AH` | 0.536 | 0.036 | 18/25 |
| the other 22 | 0.47–0.55 | not significant | — |

### C4: surface-fit heuristics (H2b, H3b, H4a)
- None passes, with either h version. The Holm-significant ones (H3b `F_AH` 0.542, H2b `F_AH` 0.537, H2b `F_AV` 0.531) are
  below a trivial baseline (`F_AH` 0.551, `F_AV` 0.534).

## Endgame: LLM direct pick (`TH_T80`, the last 7 placements)
With 7 stones left every remaining stone is on offer at every step, so endings from one state differ only by the policy.
States: seeds 313, 484, 717, 3 cascade walls each. Mean z: the four metrics, z in the seed's factorial `TH_T80` block.
Cascade: Wang's selection, no control. Plans and rules: `docs/plans/endgame-arena.md`, `llm-endgame-pick.md`,
`llm-endgame-lookahead.md`, `llm-endgame-llc-note.md`; tables `runs/batch/endgame-arena-th80.csv`,
`llm-endgame-pick-th80.csv`, `llm-endgame-lookahead-opponents.csv`, `llm-endgame-lookahead-th80.csv`,
`llm-endgame-llc-note-opponents.csv`, `llm-endgame-llc-note-th80.csv`.

### Free arena (9 states, 216 walls)
- Reading "room": an admissible ending better than the cascade by 0.1 SD or more in 4/9 states; cascade at or above the
  90th random percentile in 5/9.
- Mean z: cascade 0.15, low_top 0.32, llc 0.16, borda 0.01, random median −0.32; 41 of 180 random endings lose a stone.

### LLM pick, no lookahead (step 3, the same 9 states, gpt-6.1-terra, US$0.34)
- Reading "no endgame contribution shown": above the cascade with no fewer stones 5/9 (needs 7), not below low_top 4/9
  (needs 5), 24/59 decisions differ from the cascade.
- Lost stones in 2/9 states (484/13204: 16 placed, 717/14032: 17). Forcing every option from step 17 on: an option loses
  stones exactly when it leaves a remaining stone with 0 feasible poses at the next step (`llm-pick-step17-tree.csv`).
- Mean z: LLM 0.25, cascade 0.15, low_top 0.32.

### LLM pick with check_after (9 new states, US$0.47)
check_after gives, per candidate, each remaining stone's feasible poses at the next step (no scores). The free opponents
get the same table: low_top_safe and random_safe choose among the safe candidates (no remaining stone left at 0), pick_safe
takes the highest LLC among them. The table was checked against 161 forced rebuilds, all equal.

| Rule | Result | Bar |
| --- | --- | --- |
| (a) above the cascade, no fewer stones | 7/9 | ≥ 7 |
| (b) not below low_top | 5/9 | ≥ 5 |
| (c) decisions that differ from the cascade | 37/62 | ≥ 25% |
| (d) states that lose stones | LLM 0, cascade 1 | LLM ≤ cascade |
| (e) not below low_top_safe | 6/9 | ≥ 5 |
- Reading "go", at the bar: one state of (a) by +0.003 (717/04123: −0.129 against −0.132).
- check_after called in 53/62 decisions; no pick left a stone at 0.
- Mean z: pick_safe 0.28, LLM 0.23, llc 0.19, cascade 0.04, low_top −0.01, low_top_safe −0.04, random median −0.27.
  pick_safe loses no stone in the 9 new states, low_top_safe none in those nor in the 9 arena states; random loses
  stones in 33/180 walls, random_safe in 2/180 (safety looks one step ahead only).

### LLM against pick_safe (exploratory, not in the rule)
| State | LLM mean z | pick_safe mean z | LLM − pick_safe, `F_LR` z |
| --- | --- | --- | --- |
| 313/13420 | 0.773 | 0.875 | −1.58 |
| 313/14302 | 0.002 | 0.725 | −1.90 |
| 313/03241 | −0.063 | −0.285 | +1.17 |
| 484/02413 | 0.260 | −0.776 | +0.49 |
| 484/10243 | 0.874 | 0.411 | +0.61 |
| 484/01324 | −0.478 | −0.137 | −1.85 |
| 717/04123 | −0.129 | 0.700 | −2.35 |
| 717/23410 | 0.709 | 0.389 | +2.25 |
| 717/20341 | 0.139 | 0.607 | −1.95 |
- The LLM is below pick_safe in 5/9 states, and in all 5 its `F_LR` is lower by 1.6–2.4 SD (the largest metric gap in 4;
  in 313/14302 `F_AH` −2.13 is larger).
- In 14 of 62 decisions (13 of the 53 with check_after) the LLM did not take the highest LLC among the safe candidates,
  each time one LLC step (0.17) lower; in 11 of the 14 its pick had a higher NHR than every highest-LLC safe candidate. The
  prompt asks to compare all five scores and defines LLC and `F_LR` in near-identical words, but does not say that LLC
  is the per-step form of `F_LR`.
- The highest LLC is tied in 52 of 62 decisions; pick_safe then takes the first candidate, and the LLM chose another
  candidate of the tie in 32. In 484/02413 (LLM +1.04 over pick_safe) 2–9 safe candidates tie at the top in every
  decision; the LLM stayed within the tie in 6 of 7, a different one from pick_safe in 5, and gained mostly `F_AV`
  (+2.67) and `F_AH` (+0.99).

### LLC note against a control (a third set of 9 states, 2 × 9 walls, US$0.94)
The note is one paragraph of the system prompt: LLC and `F_LR` are the same kind of quantity (no numbers, nothing on the
other scores, not a rule). Control: the prompt of the batch above (pick-2), on the same states, built interleaved state by
state, the control first. Each batch was run once.

| Rule | Control (pick-2) | Note (pick-3) | Bar |
| --- | --- | --- | --- |
| (a) above the cascade, no fewer stones | 6/9 | 5/9 | ≥ 7 |
| (b) not below low_top | 9/9 | 9/9 | ≥ 5 |
| (c) decisions that differ from the cascade | 29/62 | 30/61 | ≥ 25% |
| (d) states that lose stones | LLM 0, cascade 1 | LLM 1, cascade 1 | LLM ≤ cascade |
| (e) not below low_top_safe | 4/9 | 6/9 | ≥ 5 |
| (f) not below pick_safe, no fewer stones | 7/9 | 5/9 | ≥ 5 |
| Decisions giving up the highest safe LLC | 9/62 (6 for a higher NHR) | 2/61 (2) | — |
| Mean z | 0.157 | 0.172 | — |
- Reading of both batches "no endgame contribution shown": both fail (a). The control repeats the prompt that read "go"
  at the bar on the previous 9 states and gets 6/9 here.
- Effect of the note: (g) the note wall above the control wall with no fewer stones in 6/9 states (needs 7), (h) fewer
  decisions giving up the highest safe LLC (2 against 9): "the LLM picks differently, the wall not shown better". Mean
  of the 9 differences in mean z: +0.015.
- By metric (exploratory), note − control in z: `F_LR` +0.63 (higher in 5 states, equal in 1, lower in 3), `F_SF` −0.33,
  `F_AH` −0.08, `F_AV` −0.16. Mean `F_LR` z: note 0.85, control 0.22, pick_safe 0.70; `F_LR` not below pick_safe's in
  7/9 states with the note, 3/9 without. The note moves the picks towards LLC and `F_LR` rises, the other three fall,
  the mean stays.
- (f) is easy on these states: pick_safe's mean z is 0.02 here (0.28 on the previous 9), at the cascade's level (−0.01)
  and above the cascade in 4/9. Opponents' mean z: borda 0.21, low_top_safe 0.12, llc 0.05, pick_safe 0.02, cascade
  −0.01, random_safe median −0.23, random median −0.40, low_top −0.51.
- With the note the LLM chose another candidate of the top-LLC tie than pick_safe in 41 of 53 tied decisions (control
  27 of 52). One note wall lost a stone (484/40321, 17 placed; the cascade loses one in another state).

| State | Note mean z | Control mean z |
| --- | --- | --- |
| 313/12403 | −0.567 | 0.126 |
| 313/41023 | 0.787 | 0.353 |
| 313/04231 | 0.347 | 0.016 |
| 484/43102 | −0.147 | −0.556 |
| 484/40321 | −0.058 | 0.566 |
| 484/21430 | 0.376 | 0.091 |
| 717/14302 | 0.409 | 0.233 |
| 717/13204 | 0.443 | 0.913 |
| 717/41203 | −0.044 | −0.330 |
