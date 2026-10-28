"""Frozen WAFT feature repair: model, checked datasets, fitting and inference."""

__version__ = "0.2.0"


def __getattr__(name):
    if name in ("MotionFeatureRepair", "MotionRepairConfig"):
        from .motion import model
        return getattr(model, name)
    if name == "FeatureRepair":
        from .model import FeatureRepair
        return FeatureRepair
    raise AttributeError(name)
