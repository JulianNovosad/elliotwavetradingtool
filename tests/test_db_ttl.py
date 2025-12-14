import pytest
import sqlite3
import datetime
import time
import os
import asyncio

# --- Mock Database and Cleanup Logic ---
# This test file will simulate the SQLite database operations and the TTL cleanup.

# Assume DB path is configurable, like from config.yaml
DB_PATH = "./data/test_prices.db"
DB_TTL_HOURS = 8

async def setup_test_db():
    """Sets up a fresh SQLite database for testing."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to DB and create table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS prices;")
    cursor.execute("""
        CREATE TABLE prices (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            price REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        );
    """)
    conn.commit()
    conn.close()
    # print(f"Test DB setup complete at {DB_PATH}")

async def insert_price_data(symbol: str, timestamp: datetime.datetime, price: float, volume: float):
    """Inserts a single price data row into the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Use ISO format for timestamp to store in TEXT column
    timestamp_iso = timestamp.isoformat()
    try:
        cursor.execute(
            "INSERT INTO prices (symbol, timestamp, price, volume) VALUES (?, ?, ?, ?)",
            (symbol, timestamp_iso, price, volume)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Handle potential duplicate primary key, though not expected in test setup
        pass
    finally:
        conn.close()

async def get_all_price_data(symbol: str = None) -> list:
    """Retrieves all data from the prices table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if symbol:
        cursor.execute("SELECT symbol, timestamp, price, volume FROM prices WHERE symbol = ? ORDER BY timestamp", (symbol,))
    else:
        cursor.execute("SELECT symbol, timestamp, price, volume FROM prices ORDER BY symbol, timestamp")
    rows = cursor.fetchall()
    conn.close()
    # Convert timestamp strings back to datetime objects for easier comparison
    return [(r[0], datetime.datetime.fromisoformat(r[1]), r[2], r[3]) for r in rows]

# --- TTL Cleanup Logic (simplified for testing) ---
async def cleanup_old_data(ttl_hours: int):
    """
    Deletes rows older than ttl_hours from the prices table.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Calculate cutoff time
    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=ttl_hours)
    cutoff_iso = cutoff_time.isoformat()
    
    # Delete rows where timestamp is older than cutoff_iso
    # Note: SQLite stores TEXT dates, so comparison is string-based. ISO format ensures this works.
    cursor.execute("DELETE FROM prices WHERE timestamp < ?", (cutoff_iso,))
    
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count

# --- Test Cases ---

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_db():
    """Sets up and tears down the test database for the entire module."""
    # Setup
    print("\nSetting up test database...")
    asyncio.run(setup_test_db()) # Use asyncio.run for async setup function
    
    yield # Run tests
    
    # Teardown
    print("\nTearing down test database...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    print("Test database removed.")

@pytest.fixture(name="db_path")
def _db_path():
    return DB_PATH

def test_db_insertion_and_retrieval(db_path):
    """Tests basic insertion and retrieval of data."""
    symbol = "BTCUSDT"
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Insert some data
    asyncio.run(insert_price_data(symbol, now - datetime.timedelta(minutes=5), 70000.50, 1000))
    asyncio.run(insert_price_data(symbol, now - datetime.timedelta(minutes=3), 70010.20, 1200))
    asyncio.run(insert_price_data(symbol, now - datetime.timedelta(minutes=1), 70020.00, 1100))
    
    # Retrieve data
    all_data = asyncio.run(get_all_price_data(symbol=symbol))
    
    assert len(all_data) == 3
    # Check timestamp conversion and order
    assert all_data[0][1] == (now - datetime.timedelta(minutes=5))
    assert all_data[1][1] == (now - datetime.timedelta(minutes=3))
    assert all_data[2][1] == (now - datetime.timedelta(minutes=1))
    assert all_data[0][2] == 70000.50

def test_db_ttl_cleanup():
    """
    Tests the TTL cleanup functionality.
    Inserts data with varying ages and checks if older data is removed.
    """
    symbol = "ETHUSD"
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Insert data:
    # 1. Very old data (should be deleted)
    old_ts = now - datetime.timedelta(hours=DB_TTL_HOURS + 2) # Older than TTL
    asyncio.run(insert_price_data(symbol, old_ts, 3000.00, 5000))
    
    # 2. Data exactly at TTL boundary (should be kept, or very close boundary check)
    # SQLite timestamps are precise, let's insert data just within TTL to be sure it's kept.
    ttl_boundary_ts = now - datetime.timedelta(hours=DB_TTL_HOURS - datetime.timedelta(minutes=1).total_seconds() / 3600)
    asyncio.run(insert_price_data(symbol, ttl_boundary_ts, 3050.00, 5500))
    
    # 3. Recent data (should be kept)
    recent_ts = now - datetime.timedelta(hours=1)
    asyncio.run(insert_price_data(symbol, recent_ts, 3100.00, 6000))
    
    # Get all data before cleanup
    data_before = asyncio.run(get_all_price_data(symbol=symbol))
    assert len(data_before) == 3, f"Expected 3 rows before cleanup, found {len(data_before)}"

    # Perform cleanup
    deleted_count = asyncio.run(cleanup_old_data(ttl_hours=DB_TTL_HOURS))
    
    # Expecting 1 row to be deleted (the 'old_ts' data)
    assert deleted_count == 1, f"Expected 1 row deleted, but {deleted_count} were deleted."

    # Get all data after cleanup
    data_after = asyncio.run(get_all_price_data(symbol=symbol))
    
    assert len(data_after) == 2, f"Expected 2 rows after cleanup, found {len(data_after)}"
    
    # Check that the deleted row is gone and others remain
    timestamps_after = [d[1] for d in data_after]
    assert old_ts not in timestamps_after
    assert ttl_boundary_ts in timestamps_after
    assert recent_ts in timestamps_after
    
    print("test_db_ttl_cleanup passed.")

# To run these tests:
# 1. Ensure pytest is installed: pip install pytest
# 2. Save this file as test_db_ttl.py in the tests/ directory.
# 3. Run pytest from the project root: pytest
# Note: This test relies on direct file system access for the DB.