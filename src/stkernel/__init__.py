"""
Structural Topology Kernel (STK) — 50 deterministic, training-free structural
coordinates for microstructure images and 2-D fields.

    import stkernel as stk
    c = stk.measure("micrograph.tif")        # dict of 50 coordinates
    df = stk.measure_many("micrographs/")    # one row per image

Part of Parallax Metrology: https://www.parallaxmetrology.com
"""
from .core import COORDINATES, VERSION, measure, measure_field, measure_many, prepare_micrograph
from .featurizer import STKFeaturizer

__version__ = "0.1.0"
__all__ = ["measure", "measure_field", "measure_many", "prepare_micrograph", "STKFeaturizer",
           "COORDINATES", "VERSION", "__version__"]
