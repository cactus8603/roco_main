# Action-bank validation protocol

Date: 2026-10-05

Status: `FROZEN_DESIGN_NOT_YET_EXECUTED`

## Short answer: is ordinary K-fold appropriate?

K-fold is the right basic idea, but ordinary row-wise K-fold is not valid for
this panel. The 1,200 rows are repeated observations of 30 physical scenes
(clean plus related corruptions), not 1,200 independent samples. Randomly
splitting rows lets the same scene appear in training and evaluation and makes
the interval much too confident.

The minimum acceptable version is nested `GroupKFold`:

| Question | Required split |
|---|---|
| Can the frozen router generalize to unseen scenes/components? | outer grouped folds |
| How are normalization, model weights, thresholds, and abstention calibrated? | inner grouped folds only |
| Is one action redundant after the remaining actions can substitute for it? | full-bank versus independently retrained leave-one-action-out pipelines on the same outer rows |
| What is the uncertainty unit? | physical scene/component cluster, not frame/row |

Five outer folds and four inner folds are a practical default, not a source of
extra evidence. `K` should be reduced if a fold would contain too few
independent groups, and a whole-component holdout should replace scene-level
grouping when deployment generalization is across datasets or capture systems.
Repeated K-fold may diagnose split sensitivity, but its repeats are correlated
and must not be counted as additional independent scenes.

## Verdict on the existing E3 validation

The E278 training loop itself uses a reasonable nested grouped cross-validation
implementation:

- six outer folds evaluate held-out physical scenes;
- for each outer fold, the other five scene groups rotate as inner validation
  folds;
- normalization, target thresholds, model fitting, and routing-policy
  calibration use only the inner side;
- all clean and corrupted observations from one physical scene stay in the same
  fold;
- metrics and bootstrap intervals are aggregated by physical scene rather than
  treating 1,200 rows as independent.

That is stronger than ordinary row-wise K-fold. Row-wise K-fold would leak the
same scene and its corruption variants across train and validation sets.

It is nevertheless **not final validation**. The complete E3 panel was used to
form the 29-to-10 capacity shortlist, then its outer-OOF outcomes were inspected
again after E275, E276, E277, and E278 to prune the bank. Repeatedly adapting the
candidate set to those outer results makes E3 an opened development panel. The
last E278 OOF estimate is therefore optimistic for the selected four-control
bank even though each individual E278 training run is nested correctly.

The existing routed-contribution audit is also insufficient for an overlap
claim. It asks whether an action's selected rows have positive net gain. It does
not ask whether the full policy is better than a retrained policy in which that
action is absent and another retained action may substitute for it.

## Required fresh-panel design

Freeze these four candidates before opening new outcomes:

1. Gaussian sigma 1.0;
2. Gaussian sigma 2.0;
3. unsharp amount 1.5 at sigma 1.0;
4. joint percentile normalization 1/99.

Native is always action zero. Operator parameters, endpoints, support, features,
model architecture, seeds, candidate grids, gates, fold assignment algorithm,
and tie breaks must be hash-bound before training.

Use nested **grouped** K-fold, not ordinary K-fold:

```text
fresh independent scenes/components
  outer GroupKFold: final OOF evaluation only
    inner GroupKFold: normalization, fitting, hyperparameters,
                      policy thresholds, and all model selection
```

Recommended default is five outer folds and four inner folds. If several scenes
come from the same capture sequence or benchmark component, that larger unit is
the group. All frames, endpoints, clean versions, corruption types, severities,
and repeats from one group stay together. Balance fold composition by benchmark
component and corruption coverage without splitting a group. The number of
independent groups must be set by a prospective power analysis; repeating folds
does not turn 30 scenes into more independent evidence.

There are two different estimands and they must not be mixed:

- **Fixed-bank confirmation:** keep the four candidates fixed and use inner CV
  only to train/calibrate the router. This estimates performance of the frozen
  E3-derived bank on genuinely new data.
- **Bank-selection pipeline:** repeat every outcome-dependent reduction and
  pruning step inside each outer-training split. The selected bank may differ by
  fold. This estimates the generalization of the selection procedure, not the
  performance of one globally selected bank.

For the present goal, fixed-bank confirmation is the next test. Do not prune or
replace an action after viewing any fresh outer-fold result. If the protocol is
changed, start a new version and obtain another untouched confirmation panel.

The E279 opened-development rehearsal has now executed this full-versus-four-LOO
design on E3. It confirms that the implementation is feasible and reveals the
expected capacity/routing distinction, but it does not replace the fresh panel.

E281 additionally removes the complete low-pass family (both Gaussian
strengths) and retrains. The family passes the opened-E3 simultaneous policy-
value test even though neither strength passes its individual action-level test.
Therefore final evaluation must report both leave-one-action-out and leave-one-
family-out results: action LOO tests exact-control redundancy, while family LOO
prevents mutually substitutable strengths from making a useful mechanism look
dispensable. The three-family comparison supports low-pass and unsharp routed
value; radiometry has unique oracle headroom but not routed necessity.

E280 removes the old trainer's hard-coded 1,200-row/six-fold assumptions and
checks the configurable implementation against E278. All 1,200 CTRL-FACT
decisions, utility/harm/catastrophe outputs, fold calibrations, bootstrap
intervals, and six serialized factorized models reproduce exactly. This is an
implementation-parity test only; because it reuses E3, it adds no action-
admission evidence. See the [E280 report](../../experiments/E280_dynamic_group_router_parity_v1/README.md)
and [parity artifact](../../experiments/E280_dynamic_group_router_parity_v1/PARITY.json).

The configurable primary trainer is
[`train_dynamic_group_router.py`](train_dynamic_group_router.py). For final
selection-aware admission, run it once on the full four-control protocol and
once on each of four hash-bound leave-one-out protocols. Each run must use the
same fresh group assignments and must be frozen before any outer result is
opened.

## Action admission and overlap test

Train five pipelines per outer split: the full four-action bank and four
leave-one-action-out banks. Each pipeline receives its own inner-CV training and
calibration. On the same outer rows, compute:

```text
delta_a = loss(policy without action a) - loss(full-bank policy)
```

An action is non-redundant only if its scene-clustered, one-sided simultaneous
confidence lower bound for `delta_a` is greater than zero. Use a max-T clustered
bootstrap or a predeclared Holm correction across the four actions. A positive
gain on rows to which an action happened to be routed is supporting diagnosis,
not the admission test.

Also train three leave-one-family-out pipelines. This family-level analysis is
not a substitute for action admission: it determines whether a repair direction
should remain represented when neighboring strengths substitute for each other.
If a family passes but no individual anchor passes, retain the smallest
prospectively frozen discrete strength set for further confirmation; do not
silently claim that every anchor is individually validated.

For a family that exposes more than one strength, add a direct strength-policy
test. Independently retrain the multi-strength router and every single-strength
router on identical folds. A strength-proposal claim requires the multi-
strength policy to beat each single-strength alternative with a simultaneous
scene-clustered lower bound above zero, in addition to clean/tail gates. Oracle
increment alone justifies preserving a candidate hypothesis, not claiming that
the router can select its strength.

E282 rehearses this test for Gaussian sigma 1/2. Both strengths add oracle
capacity, but the two-strength router selects native in all six folds and is
worse than the sigma-2-only router. Thus the current opened evidence rejects
the routed strength-proposal claim and authorizes neither interpolation nor
continuous strength. This failure is carried into the fresh protocol rather
than hidden by the full mixed-family router's route counts.

The full bank must also pass the predeclared global gates on outer OOF
predictions:

- corrupt macro gain versus native at least 5%;
- clean relative degradation at most 2%;
- catastrophe per intervention at most 0.5%;
- required corruption-coverage gate (the current four-condition panel cannot
  adjudicate the normative 12-of-20 gate);
- strictly positive scene-clustered lower confidence bound for corrupt gain.

Report fold-wise results, action route counts, harmful/catastrophic invocations,
and intervals even when a gate fails. Do not use individual rows as bootstrap
units.

## Final-list rule

Until the fresh protocol passes, the four controls are a frozen
opened-development candidate bank and the final validated action list is empty.
After a clean pass, admit only the actions whose selection-aware ablation test
passes; then rerun and report the final reduced bank under the same locked
evaluation lifecycle without using the confirmation outcomes for another tuning
round.

The exact nine-run execution matrix, source hashes, data prerequisites, global
gates, action/family/strength comparisons, and current readiness state are
frozen in [`FRESH_VALIDATION_WORK_PACKAGE.json`](FRESH_VALIDATION_WORK_PACKAGE.json).
Its verifier must pass before any fresh action outcome is opened. A passing
package verifier means the method is frozen; it does not mean that the absent
fresh data or action outcomes have passed.
