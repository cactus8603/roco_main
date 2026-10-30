"""Cumulative runtime budgets preserved from the reviewed native trainer."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import time


def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0

@dataclass(frozen=True)
class RuntimeBudgets:
    """Child wall time covers all separately bounded waits and active work."""
    neural_active_seconds: float = 3600
    before_context_wait_seconds: float = 7200
    running_pressure_wait_seconds: float = 7200
    overhead_seconds: float = 1200
    shutdown_margin_seconds: float = 300

    def __post_init__(self):
        _require(all(_finite_nonnegative(value) and value > 0 for value in asdict(self).values()),
                 'All runtime budgets must be finite and positive')

    @property
    def minimum_child_timeout_seconds(self):
        return sum(asdict(self).values())

    def validate_child_timeout(self, seconds):
        _require(_finite_nonnegative(seconds) and seconds >= self.minimum_child_timeout_seconds,
                 'Child timeout cannot be shorter than active + wait + overhead + shutdown budgets')
        return seconds

class ActiveClock:
    """Exclusive monotonic phases: pressure waiting never consumes active time."""
    PHASES = ('active', 'pressure_wait', 'before_context_wait', 'overhead')

    def __init__(self, budgets=None, *, clock=time.monotonic):
        self.budgets = budgets or RuntimeBudgets()
        self._clock = clock
        self.started = self._last = float(clock())
        _require(_finite_nonnegative(self.started), 'Finite monotonic start required')
        self.phase = 'overhead'
        self.elapsed = dict.fromkeys(self.PHASES, 0.)

    @classmethod
    def from_snapshot(cls, snapshot, *, clock=time.monotonic):
        """Resume cumulative phase budgets; downtime consumes no neural time.

        Absolute monotonic stamps cannot cross a reboot/process boundary. Only
        validated cumulative durations are restored; the new session begins in
        overhead until its caller explicitly enters a work/wait phase.
        """
        _require(isinstance(snapshot, dict)
                 and snapshot.get('schema') == 'waft-repair-active-clock/v1'
                 and isinstance(snapshot.get('elapsed_seconds'), dict)
                 and set(snapshot['elapsed_seconds']) == set(cls.PHASES)
                 and all(_finite_nonnegative(value)
                         for value in snapshot['elapsed_seconds'].values()),
                 'Invalid saved runtime phase durations')
        budgets = RuntimeBudgets(**snapshot['budgets'])
        value = cls(budgets, clock=clock)
        value.elapsed = {phase: float(snapshot['elapsed_seconds'][phase]) for phase in cls.PHASES}
        value.check()
        return value

    def _accrue(self):
        now = float(self._clock())
        _require(math.isfinite(now) and now >= self._last, 'Monotonic phase clock moved backward')
        self.elapsed[self.phase] += now - self._last
        self._last = now

    def switch(self, phase):
        _require(phase in self.PHASES, 'Unknown runtime clock phase')
        self._accrue()
        self.phase = phase

    @contextmanager
    def measure(self, phase):
        previous = self.phase
        self.switch(phase)
        try:
            yield self
        finally:
            self.switch(previous)

    def snapshot(self):
        self._accrue()
        return dict(schema='waft-repair-active-clock/v1', phase=self.phase,
                    started_monotonic=self.started, heartbeat_monotonic=self._last,
                    wall_elapsed_seconds=sum(self.elapsed.values()),
                    current_session_elapsed_seconds=self._last - self.started,
                    elapsed_seconds=dict(self.elapsed), budgets=asdict(self.budgets))

    def check(self):
        value = self.snapshot()
        limits = dict(active=self.budgets.neural_active_seconds,
                      pressure_wait=self.budgets.running_pressure_wait_seconds,
                      before_context_wait=self.budgets.before_context_wait_seconds,
                      overhead=self.budgets.overhead_seconds)
        for phase, limit in limits.items():
            _require(self.elapsed[phase] <= limit, f'{phase} runtime budget exhausted; no automatic retry')
        return value
