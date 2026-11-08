# Step-wise quality of the gpt-001 counterfactual walls

Offline, from the records of eight walls (seed 66794912, thangavelu-noise80); no wall was built, no
API was called. Generator: `research/llm_walls/step_quality.py` (definitions in its docstring and in
`summary.json`). n = 1 seed, one realisation per arm; everything here is exploratory.

| wall | role |
|---|---|
| GPT | gpt-001 (live GPT order controller) |
| Paper | dev Paper wall; also P6 |
| V | GPT's nine recorded orders replayed with `--switch`; a check, not a sample |
| G6, G8, G17 / P8, P17 | GPT's orders through step k / through step k-1, Paper order afterwards |

Files: `steps.csv` (one row per wall and placement), `pairs.csv` (D(n) = a - b per placement),
`summary.json` (per pair and metric: D at the decision, after it, at the end, first n where D != 0),
`pair_*.png`, `f_lr.png`.

## Metrics after each placement n

- **F_SF**: 03's stone filling (right-cropped area). **F_SF_progress**: stone pixels over the whole
  02 wall (H·W − 2H − 3W, the same for every wall and step).
- **F_AH_partial, F_AV_partial**: 03 unchanged on the unfinished wall. 03 crops to the highest stone,
  so the pockets above lower stones count as joints. **They are partial values, never a final
  quality**; only at n = 18 are they F_AH and F_AV.
- **lambda_left, lambda_right**: Wang's two tilting-table solves of the partial simulation wall
  (unrounded); **F_LR** = min / 0.58, the limit of the wall as it stands after n.

## Checks (all passed; the generator writes nothing if one fails)

1. V's 18-step series equals gpt-001's in every column.
2. At n = 18 every wall's F_SF, F_AH, F_AV and F_LR equal its wang-quality-v1 values exactly, and
   both lambdas equal the run's mechanism files.
3. At all 144 placements, round(lambda_min, 1) / 0.58 equals the LLC of the candidate Wang chose.
4. In every G_k / P_k pair all metrics are identical before the decision placement.

## Results: D = G − P (GPT's candidate − Paper's candidate at step k)

Directions: F_SF, F_AV, F_LR higher is better; F_AH lower is better.

| pair (decision n) | metric | at the decision | after it | final | first D ≠ 0 |
|---|---|---|---|---|---|
| k6 (n 6) | F_SF_progress | −0.0205 | +0.0037 | −0.0168 | n 6 |
| | F_AH_partial | +0.29 | +1.57 | +1.86 | n 6 |
| | F_AV_partial | +1.16 | −5.81 | −4.65 | n 6 |
| | F_LR | +0.126 | −0.001 | +0.125 | n 6 |
| k8 (n 8) | F_SF_progress | −0.0201 | +0.0206 | +0.0005 | n 8 |
| | F_AH_partial | +0.22 | −0.59 | −0.37 | n 8 |
| | F_AV_partial | +4.05 | −2.62 | +1.43 | n 8 |
| | F_LR | +0.016 | −0.031 | −0.015 | n 8 |
| k17 (n 16) | F_SF_progress | −0.0047 | +0.0048 | +0.0001 | n 16 |
| | F_AH_partial | 0 | −0.76 | −0.76 | n 17 |
| | F_AV_partial | 0 | +1.11 | +1.11 | n 17 |
| | F_LR | −0.154 | +0.690 | +0.536 | n 16 |
| GPT − Paper (n 6) | F_LR | +0.126 | +0.178 | +0.304 | n 6 |

"After it" is D(18) − D(decision): bookkeeping, not a causal split (see limits).

### At the decision placement

- Every pair differs at the decision itself; there is no delayed onset except F_AH/F_AV at k17,
  where the different stone at n 16 left 03's five shortest joint paths unchanged.
- GPT's candidate was the smaller stone each time (stone px −9188, −9031, −2119), so F_SF_progress
  starts lower by construction.
- F_LR at the decision: k6 +0.126 (both lambdas higher), k8 +0.016, k17 −0.154 (GPT's candidate
  lowered lambda_left; Paper ranks LLC first and took the higher one).

### Along the trajectory

- The partial-wall F_LR is not monotone in any wall: a placement can raise either lambda as well as
  lower it, and the weaker side changes along the way (GPT: right at n 1–4, tied at 0.58 for n 1–2, left 5–7, right 8–13,
  left 14–17, right at 18).
- k6: G6's F_LR stays above Paper's at every n ≥ 6 (D between +0.016 and +0.294); F_AV_partial falls
  behind at every n from 7 except n 11 (+0.36).
- k8: the sign of D(F_LR) changes five times; at n 13–15 G8 is 0.27–0.35 lower, at the end they are
  within 0.015.
- k17: D(F_LR) is −0.154 at n 16, −0.106 at n 17 and +0.536 at n 18. At n 18 both walls hold the
  same 18 stones; G17's lambda_left rises from 0.114 to 0.492 with its last placement, P17's falls
  from 0.285 to 0.062.

### Final wall

- F_SF: within 0.001 for k8 and k17, where both walls end with the same stones. k6's −0.017 comes
  from 02's overlap rule: from n 9 one of G6's stones overlaps by ≥ 200 px and is left out of the
  rendered wall (all other walls: none), which also enters G6's F_AH and F_AV.
- F_LR: k6 +0.125, k8 −0.015, k17 +0.536; GPT − Paper +0.304. The three pair differences do not add
  up to it and are not expected to.

## How not to read F_LR

The candidate LLC at step k is round(lambda_min, 1) / 0.58 of the wall after that placement, so the
final F_LR and the last LLC measure the same wall. Their agreement is an identity, not evidence that
the last stone decides a wall's stability. The final lambda belongs to the whole 18-stone
configuration: in k17 the last placements of the two walls differ in position and order with the
same stone set, and which mechanism governs depends on everything below them. This analysis does not
attribute F_LR to any single stone.

## Limits

1. One seed, one realisation; no variance estimate. D values are descriptions, not effects with
   uncertainty.
2. After the decision the walls sample and place differently; "after it" mixes the decision's
   consequences with that path dependence and cannot separate them.
3. Both arms continue with the Paper order, a policy GPT never used; the effect is conditional on
   that continuation.
4. Steps 6, 8, 17 were chosen after seeing gpt-001.
5. F_AH_partial and F_AV_partial count the unfinished top as joints; compare them between walls at
   the same n, never with final values.
6. F_SF, F_AH and F_AV follow 02's rendering, including its overlap rule (G6).
