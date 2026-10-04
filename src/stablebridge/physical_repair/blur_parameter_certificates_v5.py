"""Rejected family-specific disk-signature diagnostic over E142 evidence.

The E143 clean counterexamples were both radius-one disk actions with negative
held-out disk-vs-identity spectral evidence.  Gaussian spectral scores are
known to be biased by window leakage, so this successor does not promote a
generic spectrum threshold.  It adds only the family-specific necessary
condition that a directly delivered disk hypothesis must improve its own OTF
model over identity on the independent B fold.  A known synthetic radius-six
disk counterexample also has negative spectral LCB because of window leakage,
so this rule is retained as a negative test and is not delivery-ready.
"""
from __future__ import annotations

from .blur_parameter_certificates_v3 import action_specific_direct_winners


MECHANISM_STATUS = "REJECTED_TRUE_DISK_COUNTEREXAMPLE"


def action_specific_direct_winners_v5(evidence) -> tuple[str, ...]:
    """Apply a zero-extra-cost disk OTF own-null gate to v3 direct actions."""
    output = []
    for key in action_specific_direct_winners(evidence):
        value = evidence[key]
        if (value.family == "disk"
                and value.own_null_spectral_improvement_lcb <= 0.0):
            continue
        output.append(key)
    return tuple(output)
