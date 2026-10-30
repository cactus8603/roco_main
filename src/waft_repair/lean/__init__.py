"""Encoder-only LiteR before one frozen five-update WAFT consumer."""
__all__ = ["LiteFeatureRepair", "LiteRepairConfig", "LeanSystem", "make_system",
           "make_optimizer", "train_step", "evaluate"]


def __getattr__(name):
    if name in ('LiteFeatureRepair', 'LiteRepairConfig'):
        from . import lite_repair
        return getattr(lite_repair, name)
    if name in __all__:
        from . import runtime
        return getattr(runtime, name)
    raise AttributeError(name)
