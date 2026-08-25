"""Built-in dataset adapters."""

from histodata.adapters.kather100k import Kather100KAdapter, Kather100KNonNormAdapter
from histodata.adapters.pcam import PCamAdapter
from histodata.adapters.spider import (
    SpiderBreastAdapter,
    SpiderColorectalAdapter,
    SpiderSkinAdapter,
    SpiderThoraxAdapter,
)

__all__ = [
    "Kather100KAdapter",
    "Kather100KNonNormAdapter",
    "PCamAdapter",
    "SpiderBreastAdapter",
    "SpiderColorectalAdapter",
    "SpiderSkinAdapter",
    "SpiderThoraxAdapter",
]
