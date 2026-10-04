import pytest

from stablebridge.bridge_runtime import (
    ActionSpec,
    FLOW_PAIR2,
    STEREO_CORE3,
    pairbridge_da_decision,
    plan_direct_execution,
    stablebridge_s0_decision,
    validate_registry,
)


def test_both_bridge_adapters_produce_one_path():
    stereo = plan_direct_execution(stablebridge_s0_decision("s", 2), STEREO_CORE3)
    flow = plan_direct_execution(pairbridge_da_decision("f", 0), FLOW_PAIR2)
    assert (stereo.selected_action_id, stereo.executor_id, stereo.expert_forwards) == (
        "sharpen", "croco_stereo", 1)
    assert (flow.selected_action_id, flow.executor_id, flow.expert_forwards) == (
        "cc", "waft_cc", 1)


def test_registry_and_action_mismatch_fail_closed():
    with pytest.raises(ValueError):
        validate_registry((ActionSpec("a", "x", "x"),))
    with pytest.raises(ValueError):
        plan_direct_execution(stablebridge_s0_decision("s", 1), FLOW_PAIR2)
    with pytest.raises(ValueError):
        stablebridge_s0_decision("s", 3)
