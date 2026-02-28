"""Core algorithm modules for GlobalMarketAnalyzer."""
import sys
from pathlib import Path

# Add project root to sys.path so intra-package imports work
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.graph_builder import GraphBuilder
from core.heat_engine import HeatEngine
from core.capital_field import CapitalField
from core.fundamental_filter import FundamentalFilter
from core.inertia_detector import InertiaDetector

try:
    from core.regime_calibrator import RegimeCalibrator
except ImportError:
    RegimeCalibrator = None  # module is function-based, no class

__all__ = [
    "GraphBuilder", "HeatEngine", "CapitalField",
    "FundamentalFilter", "InertiaDetector", "RegimeCalibrator",
]
