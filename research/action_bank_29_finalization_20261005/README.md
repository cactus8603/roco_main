# 29-arm action-bank reduction

Date: 2026-10-05

Status: `FROZEN_4_CONTROL_CANDIDATE_FINAL_VALIDATED_BANK_EMPTY`

## Decision

The 29-entry E243 manifest remains the provenance catalog. It is not the
runtime bank. Capacity reduction first produced four mechanism families with
ten exact-control anchors. Nested routing diagnostics then reduced that set to
the following frozen candidate bank, plus native:

| Family | Exact anchors |
|---|---|
| low-pass HF suppression | Gaussian sigma 1.0, 2.0 |
| detail recovery | unsharp amount 1.5 |
| joint radiometry | joint percentile 1/99 |

The four controls are not yet validated final actions. E275--E278 repeatedly
used the same E3 outer-OOF outcomes to prune controls, so the panel became
opened development evidence despite the nested grouped training within each
individual run. The final validated bank therefore remains empty.

## Evidence

The 15 optical/matcher controls were joined on the hash-bound 1,200-row,
30-scene opened E3 development panel. Exhaustive subset search used native as a
positive-only rollback option and required:

- at least 99.5% of the full 15-control oracle capacity overall;
- at least 98% in each of brightness, Gaussian blur, Gaussian noise, and JPEG;
- at least one exact control from every theoretically distinct family.

The smallest capacity set has ten controls. It retains 99.523224% overall and
at least 98.433115% per corruption. Removing any one of the ten controls makes
the frozen set fail at least one reduction gate. A 2,000-draw scene-cluster
bootstrap for the fixed subset gives a 99.079549%–99.770678% 95% interval for
overall retention. This interval does not correct for subset-selection
optimism and is not fresh qualification.

E275--E278 then retrained the observable router after each action-level screen:

```text
29 catalog -> 10 capacity anchors -> 7 -> 6 -> 4 frozen candidates
```

The four-candidate E278 run had 5.3547% corrupt macro gain, 0.5023% clean
degradation, improvement in all four available corruptions, and a 0.3289%
catastrophe/intervention rate. Its scene-bootstrap 95% intervals were
1.6583%--9.1815% for corrupt gain and 0.2095%--0.8172% for clean degradation.
Every retained action had a positive routed-contribution lower bound. These are
useful development results, but the sequential reuse of E3 prevents them from
being final validation.

E279 then implements the stronger overlap test: independently retrain the full
bank and each leave-one-action-out bank under the same nested grouped folds. All
four controls have positive Bonferroni-simultaneous oracle-capacity lower bounds,
so none is a repair-ceiling duplicate. For routed policy value, only unsharp
amount 1.5 has a positive simultaneous lower bound. Removing joint percentile
1/99 raises corrupt macro gain from 5.3547% to 6.0785%, but loses the small
brightness improvement. Gaussian sigma 1 and 2 have positive point effects but
their simultaneous intervals cross zero. This is exactly why the four remain a
frozen *candidate* set rather than a final list: the mechanisms add capacity,
while fresh data must decide whether their capacity is observably routable.
An E3-variance planning approximation gives 65, 136, and 607 independent scenes
for unsharp 1.5, Gaussian sigma 1, and Gaussian sigma 2 respectively at 80%
power with one-sided `alpha=0.05/4`; the radiometry routed point effect is
negative. These counts are not guarantees, but they make clear that another
30-scene panel would still be underpowered for per-action admission.

E281 adds the missing family-level overlap test. Removing sigma 1 and sigma 2
together, then retraining the remaining unsharp/radiometry router, increases
corrupt macro EPE by 0.022499 px; the one-sided three-family simultaneous lower
bound is +0.000348 px. Thus the two Gaussian strengths are individually
substitutable and underpowered, but the low-pass family as a whole has supported
routed value. Unsharp also passes at family level (+0.006960 px lower bound),
whereas radiometry still fails routed necessity despite a +0.002447 px oracle-
capacity lower bound. This supports keeping two discrete low-pass anchors as a
capacity-preserving strength hypothesis while leaving radiometry explicitly
conditional on fresh brightness/coverage evidence.

For the strict primary confirmation, later selection-aware evidence narrows
the active hypothesis further to **R4 unsharp amount 1.5 only**. R4 is the only
exact action whose action-LOO simultaneous lower bound is positive. P1 and R2
remain low-pass shadow hypotheses (the two-strength routed proposal failed),
and P3 remains a radiometry shadow hypothesis (its routed point effect is
negative). The frozen four-action matrix is retained for optional mechanism
diagnostics, not as the active admission list. See
[PRIMARY_CONFIRMATION_SHORTLIST.json](PRIMARY_CONFIRMATION_SHORTLIST.json).

E282 then tests the strength claim directly by independently retraining
native-plus-{sigma 1, sigma 2}, native-plus-sigma 1, and native-plus-sigma 2
routers. The two-strength oracle gains 9.2731% on corruptions and both anchors
have positive simultaneous oracle increments, but the two-strength learned
router calibrates to native in all six folds and gets zero routed gain. Sigma 2
alone obtains 2.2268% gain with 17 interventions. Therefore complementary
capacity is established, but the present router's strength proposal is
**rejected** on opened E3. The two anchors remain candidates only so a fresh
test can decide whether a better-observed strength policy recovers that ceiling;
continuous strength remains unauthorized.

See [E279](../../experiments/E279_selection_aware_loo_v1/README.md) and the
[action-level result](E279_SELECTION_AWARE_LOO_RESULT.json), plus the
[E281 family-level report](../../experiments/E281_selection_aware_family_loo_lowpass_v1/README.md)
and [combined family result](E281_SELECTION_AWARE_FAMILY_LOO_RESULT.json), and
the [E282 strength-proposal report](../../experiments/E282a_lowpass_two_strength_router_v1/README.md).

Every one of the original 29 entries now has an explicit disposition in the
[admission matrix](ACTION_ADMISSION_MATRIX.json): 4 frozen candidates, 6
observable-routing rejections, 5 capacity-overlap rejections, 14 shadow actions,
and 0 final validated actions.

The theoretical grouping follows the E248 mechanism audit:

- Gaussian strengths share one heat-kernel/scale-space path but are not
  scalar-equivalent responses.
- Unsharp strengths share `x + a(x - G(x))` and form one dose path.
- The 1/99 radiometric blends lie on one affine delivery ray; scalar,
  per-channel, and 5/95 variants remain distinct operator variants.
- Matcher iterations form one compute axis and do not alter the image.
- Learned models for the same task are comparators inside one mechanism group,
  not separate active families before a matched Pareto test.

## Reproduce and verify

The E275 native-uncertainty/action diagnostic can be recomputed independently
of the eventual final bank. It verifies the frozen protocol, teacher, feature
schema, and feature-record hashes before joining the ten exact controls to the
fourteen target-free `B0_native` uncertainty scalars. For every exact control it
reports uncertainty-versus-gain and uncertainty-versus-harm Spearman
associations, benefit and severe-harm AUROC, and physical-scene bootstrap
intervals:

```bash
TMPDIR=/ssd1/cactus8603/roco_main/.runtime_tmp \
PYTHONDONTWRITEBYTECODE=1 \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python -B \
  research/action_bank_29_finalization_20261005/analyze_e275_uncertainty_action.py \
  --bootstrap-draws 2000 --write

TMPDIR=/ssd1/cactus8603/roco_main/.runtime_tmp \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python -B -m unittest \
  tests.stablebridge.test_e275_uncertainty_action_diagnostic
```

This diagnostic is deliberately development-only. E3/E275 outcomes were
opened and adaptively reused, no multiplicity correction is applied, and the
stored scalar EPE outcomes cannot reconstruct pixel benefit/harm mass, harmed
fraction, or pixel CVaR. Its report grants no final-bank, selector-admission,
calibration, safety, scientific, or production authority. The generated result
is [E275_UNCERTAINTY_ACTION_DIAGNOSTIC.json](E275_UNCERTAINTY_ACTION_DIAGNOSTIC.json).

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python \
  research/action_bank_29_finalization_20261005/analyze_optical_reduction.py --write

PYTHONDONTWRITEBYTECODE=1 python3 \
  research/action_bank_29_finalization_20261005/verify_reduced_candidate_bank.py
```

The first command verifies all source hashes before recomputing the capacity
result. The second proves that the 29 source controls are partitioned exactly
once into 10 capacity-shortlist, 5 overlap-pruned, and 14 shadow controls.

The frozen four-control candidate and empty final list are checked with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python \
  research/action_bank_29_finalization_20261005/verify_opened_candidate_bank.py

PYTHONDONTWRITEBYTECODE=1 \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python \
  research/action_bank_29_finalization_20261005/verify_selection_aware_loo.py

PYTHONDONTWRITEBYTECODE=1 \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python \
  research/action_bank_29_finalization_20261005/verify_selection_aware_family_loo.py

PYTHONDONTWRITEBYTECODE=1 \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python \
  research/action_bank_29_finalization_20261005/verify_strength_proposal.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  research/action_bank_29_finalization_20261005/verify_action_admission_matrix.py
```

See [the validation protocol](VALIDATION_PROTOCOL.md) for the audit of the
current K-fold design and the required fresh nested grouped evaluation.
The [fresh-validation work package](FRESH_VALIDATION_WORK_PACKAGE.json) retains
the nine-run four-action matrix as an optional comprehensive diagnostic, while
the primary admission path is now one predeclared nested-grouped R4-versus-native
run. It remains explicitly not ready until an untouched powered panel,
before-only features, and exact action outcomes are available.
GPU availability is no longer a blocker: E283 ran native plus the four exact
controls on one already-opened E3 row on an RTX 3090 and verified finite
full-resolution outputs. CUDA is visible only outside the filesystem sandbox,
so execution requires that boundary. E283 is an execution-only receipt, not
fresh evidence and not action-admission authority; see the
[E283 report](../../experiments/E283_action_bank_gpu_preflight_v1/README.md).

The previously stale acquisition status has also been reconciled. E284 binds
the completed opaque TartanAir acquisition into a fresh five-fold grouped
panel: 20 environments, 160 physical pairs, and 9,760 clean/Robust20 rows,
with zero overlap against the eight official members decoded earlier. No
current flow payload was decoded during this freeze. Since 20 independent
environment groups are below the predeclared minimum of 65, E284 is
reject-only: it can falsify candidates but cannot positively admit one by
itself. See the
[E284 report](../../experiments/E284_tartanair_external_panel_freeze_v1/README.md).
E285 has now completed the R4-only target-free phase for all 9,760 rows:
native/R4 dense predictions are content-hashed, all 60 before-only features are
retained, exact cross-invocation reproduction passes, and no official-flow path
appears in a receipt. This removes feature/inference readiness as a blocker but
does not grant outcome decode, rejection, or admission authority; see the
[E285 report](../../experiments/E285_tartanair_r4_preoutcome_inference_v1/README.md).
E286 freezes the corresponding outcome scorer and reject-only rules. It
fail-closes before model initialization or output creation unless a separate
authorization artifact binds the protocol and E285 receipt. Futility, clean
harm, or catastrophe evidence may reject R4; a positive result can only survive
phase 1 and still requires powered confirmation. See the
[E286 report](../../experiments/E286_tartanair_r4_reject_only_scoring_v1/README.md).
The [powered-panel options audit](POWERED_PANEL_OPTIONS_AUDIT.json) also checks
the still-sealed KITTI H2 cohort using metadata only. Its 70 independent scenes
meet the smallest 65-group planning threshold, but not the 136/607-group
thresholds. Its existing E186 contract also has only five corruptions rather
than the required 20, and its current execution authority is false. H2 therefore
remains an option requiring explicit authorization and a separate pre-outcome
action-bank protocol with normative coverage, not evidence that can be counted
now.

The generalized grouped trainer has also been checked for exact backward
parity. E280 reproduces every one of the 1,200 E278 CTRL-FACT decisions and all
three output arrays with zero difference; fold calibration, bootstrap
intervals, and all six factorized model hashes are also identical:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python \
  research/action_bank_29_finalization_20261005/analyze_dynamic_group_router_parity.py

PYTHONDONTWRITEBYTECODE=1 \
  /ssd7/cactus8603/roco_spring/.conda-stereo/bin/python \
  research/action_bank_29_finalization_20261005/verify_fresh_validation_work_package.py
```

See the [E280 report](../../experiments/E280_dynamic_group_router_parity_v1/README.md).
This proves the arbitrary-`K` implementation preserves the old six-fold
behavior; it does not make E3 fresh or add admission authority.

## Remaining admission work

The four controls must be frozen and evaluated on a genuinely fresh panel with
nested grouped K-fold. All outcome-dependent fitting and policy calibration
belong in the inner loop. Action necessity must be tested by retraining the full
bank and each leave-one-action-out bank within every outer split, using a
simultaneous scene-clustered interval across the four comparisons. The E274
package does not qualify these E3-derived candidates. No candidate moves into
the final validated bank until the fresh gates pass.
