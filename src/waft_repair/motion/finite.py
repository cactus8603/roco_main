"""Finite-value safety checks with one scalar observation per device.

No model arithmetic or coverage changes: every tensor gets the original
``torch.isfinite(tensor).all()`` predicate. Only scalar observation is grouped
by device, so the caller synchronizes with each device once per boundary.
CPU AdamW step scalars stay on CPU. Importing this module initializes no CUDA.
"""
from collections import defaultdict

import torch


def all_tensors_finite(tensors):
    """Return whether every element of every input tensor is finite.

    Accepts a one-shot iterable, including a generator. Empty input returns
    True; the caller retains any separate missing-gradient rejection. All
    tensors are evaluated even when an earlier tensor is nonfinite. Each
    device's scalar predicates are reduced locally before one Python bool.
    There are no implicit device copies, persistent caches or tensor writes.
    """
    grouped = defaultdict(list)
    for tensor in tensors:
        grouped[tensor.device].append(torch.isfinite(tensor).all())
    # Enqueue every group before observing device values. Never mix CPU step
    # scalars with CUDA moment scalars in the same stack.
    reductions = [torch.stack(predicates).all() for predicates in grouped.values()]
    # Avoid all(generator)'s short circuit: every device must be observed.
    results = [bool(reduction) for reduction in reductions]
    return all(results)
