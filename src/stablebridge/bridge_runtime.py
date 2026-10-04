"""Task-neutral decision and direct-execution contract for both Bridge methods.

The contract standardizes action naming, fallback semantics, provenance, and
the one-selected-path schedule.  It deliberately does not require the two
tasks to share a learned policy or an action vocabulary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    executor_id: str
    input_variant: str
    is_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.action_id or not self.executor_id or not self.input_variant:
            raise ValueError("Action IDs, executors, and input variants must be nonempty")


@dataclass(frozen=True)
class BridgeDecision:
    task: str
    sample_key: str
    policy_id: str
    decision_source: str
    action_id: str
    fallback_action_id: str
    observer_ids: tuple[str, ...] = ()
    calibration_id: str = "frozen"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.task not in {"stereo", "flow"}:
            raise ValueError("task must be stereo or flow")
        if not all((self.sample_key, self.policy_id, self.decision_source,
                    self.action_id, self.fallback_action_id, self.calibration_id)):
            raise ValueError("Decision identity and provenance fields must be nonempty")


@dataclass(frozen=True)
class ExecutionPlan:
    task: str
    sample_key: str
    policy_id: str
    decision_source: str
    selected_action_id: str
    fallback_action_id: str
    executor_id: str
    input_variant: str
    expert_forwards: int
    observer_ids: tuple[str, ...]
    calibration_id: str

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


STEREO_CORE3 = (
    ActionSpec("identity", "croco_stereo", "observed_pair", True),
    ActionSpec("denoise", "croco_stereo", "binomial3_pair"),
    ActionSpec("sharpen", "croco_stereo", "unsharp_r2_p150_pair"),
)

FLOW_PAIR2 = (
    ActionSpec("cc", "waft_cc", "observed_pair", True),
    ActionSpec("rr", "waft_rr", "recovery_pair"),
)


def validate_registry(actions: Sequence[ActionSpec]) -> dict[str, ActionSpec]:
    if not actions:
        raise ValueError("Action registry cannot be empty")
    registry = {action.action_id: action for action in actions}
    if len(registry) != len(actions):
        raise ValueError("Action IDs must be unique")
    fallbacks = [action for action in actions if action.is_fallback]
    if len(fallbacks) != 1:
        raise ValueError("Action registry must declare exactly one fallback")
    return registry


def plan_direct_execution(decision: BridgeDecision,
                          actions: Sequence[ActionSpec]) -> ExecutionPlan:
    """Select exactly one task-native executor after a frozen Bridge decision."""
    registry = validate_registry(actions)
    if decision.action_id not in registry:
        raise ValueError(f"Unknown selected action: {decision.action_id}")
    if decision.fallback_action_id not in registry:
        raise ValueError(f"Unknown fallback action: {decision.fallback_action_id}")
    if not registry[decision.fallback_action_id].is_fallback:
        raise ValueError("Decision fallback does not match the registry fallback")
    selected = registry[decision.action_id]
    return ExecutionPlan(
        task=decision.task,
        sample_key=decision.sample_key,
        policy_id=decision.policy_id,
        decision_source=decision.decision_source,
        selected_action_id=selected.action_id,
        fallback_action_id=decision.fallback_action_id,
        executor_id=selected.executor_id,
        input_variant=selected.input_variant,
        expert_forwards=1,
        observer_ids=decision.observer_ids,
        calibration_id=decision.calibration_id,
    )


def stablebridge_s0_decision(sample_key: str, action: int,
                             metadata: Mapping[str, Any] | None = None) -> BridgeDecision:
    names = ("identity", "denoise", "sharpen")
    if action not in range(len(names)):
        raise ValueError("StableBridge S0 action must be 0, 1, or 2")
    return BridgeDecision(
        task="stereo",
        sample_key=sample_key,
        policy_id="stablebridge-s0-frozen-20260906",
        decision_source="task_native",
        action_id=names[action],
        fallback_action_id="identity",
        observer_ids=("croco-prefix-d1", "croco-prefix-d8"),
        calibration_id="stablebridge-s0-threshold-0.699835896",
        metadata=metadata,
    )


def pairbridge_da_decision(sample_key: str, action: int,
                           metadata: Mapping[str, Any] | None = None) -> BridgeDecision:
    names = ("cc", "rr")
    if action not in range(len(names)):
        raise ValueError("PairBridge action must be 0 or 1")
    return BridgeDecision(
        task="flow",
        sample_key=sample_key,
        policy_id="pairbridge-e21-e22-seed17",
        decision_source="shared_observer_task_private_head",
        action_id=names[action],
        fallback_action_id="cc",
        observer_ids=("da-clip-degradation-pair", "pairbridge-private-residual", "flow-clean-firewall"),
        calibration_id="e22-flow-q99.5",
        metadata=metadata,
    )
