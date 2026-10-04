from pathlib import Path

import pytest

from stablebridge.robust20 import ROBUSTSPRING20, validate_condition_names


def test_exact_robust20_contract_passes() -> None:
    validate_condition_names(ROBUSTSPRING20)


def test_clean_cannot_enter_robust_only_panel() -> None:
    with pytest.raises(ValueError, match="clean is forbidden"):
        validate_condition_names((*ROBUSTSPRING20, "clean"))


def test_missing_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="RobustSpring-20 mismatch"):
        validate_condition_names(ROBUSTSPRING20[:-1])
