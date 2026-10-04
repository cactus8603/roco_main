from __future__ import annotations

import stablebridge.physical_repair as physical_repair
from stablebridge.physical_repair import (
    action_bank_adapter,
    action_bank_contracts,
    child_observables,
)


PUBLIC_ACTION_MODULES = (
    action_bank_contracts,
    action_bank_adapter,
    child_observables,
)


def test_action_contract_public_names_are_reexported_by_package() -> None:
    expected_names = {
        name
        for module in PUBLIC_ACTION_MODULES
        for name in module.__all__
    }

    assert expected_names <= set(physical_repair.__all__)
    for module in PUBLIC_ACTION_MODULES:
        for name in module.__all__:
            assert getattr(physical_repair, name) is getattr(module, name)


def test_action_contract_public_names_are_unique_and_star_importable() -> None:
    expected_names = [
        name
        for module in PUBLIC_ACTION_MODULES
        for name in module.__all__
    ]
    assert len(expected_names) == len(set(expected_names))

    imported: dict[str, object] = {}
    exec("from stablebridge.physical_repair import *", {}, imported)
    assert set(expected_names) <= imported.keys()
