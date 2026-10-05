# Research TODO

## Immediate validation

- [ ] Complete the full-resolution Sintel Clean/Final comparison between the
  pretrained baseline and refined model.
- [ ] Report 1 px, 3 px, and 5 px outlier rates alongside EPE.
- [ ] Compare the selected checkpoint with the final checkpoint when they
  differ.

## Mechanism evidence

- [ ] Isolate the contribution of learned uncertainty using a matched control.
- [ ] Measure whether action-bank fine-tuning improves the refined model.
- [ ] Compare training with and without SAM-guided regional regularization.
- [ ] For SAM, report boundary error, usable-region coverage, and regional gain
  in addition to global EPE.

## Cross-backbone replication

- [ ] Train the same uncertainty and refinement components for WAFT.
- [ ] Evaluate baseline, refined, SAM-guided, and action-selected WAFT models
  under one fixed protocol.
- [ ] Keep earlier selective-repair results separate from the new training
  results.

## Statistical confidence

- [ ] Repeat the final comparison with at least three random seeds.
- [ ] Report mean, variance, and confidence intervals.
- [ ] Separate internal crop validation, public full-resolution validation,
  and hidden-test results.
