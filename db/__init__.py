"""Database and data ingestion modules for GlobalMarketAnalyzer."""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from db.database_manager import DatabaseManager
from db.fred_client import FREDClient

__all__ = ["DatabaseManager", "FREDClient"]
