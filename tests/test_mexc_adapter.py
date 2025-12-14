"""
Test suite for MEXC adapter.
"""

import pytest
import pandas as pd
import datetime
from ingest.adapters import MEXCAdapter

def test_mexc_adapter_initialization():
    """Test MEXCAdapter initialization."""
    # Test basic initialization
    adapter = MEXCAdapter(symbol="BTCUSDT", interval="1m")
    assert adapter is not None, "MEXCAdapter should initialize successfully"
    assert adapter.mexc_symbol == "BTC_USDT", "Should convert symbol format"
    assert adapter.base_url == "https://api.mexc.com", "Should have correct base URL"
    
    # Test with symbol that already has underscore
    adapter2 = MEXCAdapter(symbol="ETH_USDT", interval="5m")
    assert adapter2.mexc_symbol == "ETH_USDT", "Should keep existing underscore"


def test_mexc_interval_mapping():
    """Test MEXC interval mapping."""
    adapter = MEXCAdapter(symbol="BTCUSDT", interval="1m")
    
    # Test valid intervals
    assert adapter.map_interval_to_mexc("1m") == "1m", "Should map 1m correctly"
    assert adapter.map_interval_to_mexc("5m") == "5m", "Should map 5m correctly"
    assert adapter.map_interval_to_mexc("1h") == "1h", "Should map 1h correctly"
    assert adapter.map_interval_to_mexc("1d") == "1d", "Should map 1d correctly"
    
    # Test invalid interval
    assert adapter.map_interval_to_mexc("3m") is None, "Should return None for unsupported interval"


def test_get_interval_seconds():
    """Test conversion of interval to seconds."""
    adapter = MEXCAdapter(symbol="BTCUSDT", interval="1m")
    
    # Test various intervals
    assert adapter.get_interval_seconds("1m") == 60, "1m should be 60 seconds"
    assert adapter.get_interval_seconds("5m") == 300, "5m should be 300 seconds"
    assert adapter.get_interval_seconds("1h") == 3600, "1h should be 3600 seconds"
    assert adapter.get_interval_seconds("1d") == 86400, "1d should be 86400 seconds"
    
    # Test default for unknown interval
    assert adapter.get_interval_seconds("unknown") == 60, "Unknown interval should default to 60 seconds"


if __name__ == "__main__":
    pytest.main([__file__])