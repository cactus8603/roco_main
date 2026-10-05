# Optical Flow Research Progress

Last updated: 2026-10-06

This directory summarizes completed evidence, how to interpret it, and the
remaining research questions. Internal experiment identifiers, scheduler
details, and step-by-step engineering logs are intentionally omitted.

## Main result

The uncertainty-aware SEA-RAFT refinement reduced endpoint error from `3.1829`
to `1.1444` under the same internal validation protocol, a relative reduction
of approximately `64.0%`.

| Model state | Validation EPE | Interpretation |
| --- | ---: | --- |
| Pretrained SEA-RAFT baseline | 3.1829 | Reference under the internal protocol |
| Initial refinement stage | 3.1723 | Little improvement from initialization alone |
| Trained uncertainty-aware refinement | **1.1444** | Large gain formed during joint refinement training |

The final selected checkpoint also obtained:

- AUSE: `0.01764`
- Spearman correlation: `0.77466`
- severe-error AUROC: `0.98900`
- calibration MAE: `0.06698`

These results support that the trained model learned a useful correction under
the internal protocol. They do not yet establish superiority on the official
MPI-Sintel hidden test.

## How to interpret the result

The internal validation set contains three Sintel scenes, evaluates Clean and
Final renderings together, and uses deterministic 384×832 crops. It is useful
for model development because every compared checkpoint uses the same data and
metric, but it is not equivalent to the official full-resolution test.

The RGB and flow scales have been audited:

```text
RGB images: uint8 [0,255]
dataset and augmentation: float [0,1]
matcher boundary: float [0,255]
SEA-RAFT internal normalization: 2*(RGB/255)-1
flow prediction and ground truth: pixel displacement
```

The conversion to matcher scale occurs exactly once. The observed gain is
therefore not explained by learning a factor-of-255 input or output conversion.

The uncertainty metrics and EPE should still be interpreted separately. A
checkpoint can rank difficult pixels well while having worse flow accuracy, or
be well calibrated without producing the lowest EPE. Model selection and paper
reporting should retain all three views: accuracy, ranking, and calibration.

## Standardized evaluation

A full-resolution MPI-Sintel evaluation is in progress. It reports Clean and
Final separately, uses RGB `[0,255]`, and compares the pretrained baseline and
the refined model with the same evaluator.

The completed baseline Clean pass is currently `1.3894` EPE. The refined-model
result is the decisive next measurement; no official-performance claim should
be made before that comparison is complete.

## SAM-guided refinement

The SAM-guided variant uses segmentation regions to apply an
uncertainty-filtered regional homography regularizer. Its data preparation and
training path are operational, but no validation result is available yet.

Accordingly, there is currently no evidence that SAM improves optical flow.
The appropriate comparison is the same refined model trained with and without
the SAM regularizer, supplemented by boundary-region error and usable-region
coverage.

## WAFT cross-backbone evidence

Earlier frozen WAFT evaluation provides useful cross-backbone evidence for the
risk-localization mechanism:

| Measurement | Result |
| --- | ---: |
| Error-localization AUROC | 0.8601 |
| Error-localization AP | 0.8112 |
| Candidate-direction AUROC | 0.4844 |

The high localization score and near-random direction score support a clear
separation: disagreement identifies where the model may be wrong, but does not
identify which alternative prediction is better.

A frozen selective-repair policy on WAFT produced `+0.2819 px` clean gain and
`+0.2610 px` corrupt gain, with a scene-bootstrap 95% confidence interval of
`[0.1520, 0.4137]` for corrupt gain and `0.1489 px` worst harm.

This is evidence for the earlier selective-repair mechanism. The new
uncertainty-aware training method has not yet been replicated on WAFT, so the
two claims must remain separate.

## Current evidence boundary

Supported:

- strong improvement under one consistent internal validation protocol;
- reliable localization of severe errors;
- cross-backbone evidence that localization and action selection are different
  problems;
- positive frozen selective-repair evidence on WAFT.

Not yet supported:

- superiority on the official Sintel hidden test;
- a positive contribution from SAM-guided regularization;
- a positive contribution from action-bank fine-tuning;
- replication of the new training gain on WAFT;
- multi-seed statistical stability.
