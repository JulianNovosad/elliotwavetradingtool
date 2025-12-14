import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import datetime
import logging
import json
import threading
import sqlite3
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Local Imports ---
# Assuming these modules are in the 'analysis', 'ingest', and 'data' directories respectively.
# We might need to adjust sys.path or use relative imports if running as a package.
# For now, assume they are accessible.

# Updated imports to match the actual structure
from ingest.adapters import AdapterFactory, DataAdapter, SampleCSVAdapter, BinanceAdapter, YahooFinanceAdapter
from analysis.wave_detector import WaveDetector, ElliottRuleEngine  # Orchestrates analysis pipeline
from analysis.nms import NMS
from analysis.confidence import ConfidenceScorer

# --- Configuration ---
# Load configuration from config.yaml
try:
    import yaml
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logging.error("config.yaml not found. Using default configuration values.")
    # Provide default values if config is missing
    config = {
        "db_path": "./data/prices.db",
        "db_ttl_hours": 8,
        "default_resolution": "1m",
        "max_historical": "6 months", # String representation, will parse later
        "symbol_rate_limit_seconds": 1,
        "max_candidate_counts": 3,
        "min_wave_duration_seconds": 60, # "one wave per minute"
        "max_wave_duration_days": 7,     # "one wave per week"
        "elliott_rule_strictness": "moderate",
        "confidence_weights": {
            "rule_compliance": 0.6,
            "amplitude_duration_norm": 0.2,
            "volatility_penalty": 0.1,
            "chaos_metric": 0.1,
        },
        "count_weights": { # Weights for scoring overall wave counts
            "rule_compliance": 0.7,
            "avg_segment_conf": 0.3,
        },
        "binance_poll_interval_seconds": 60,
        "yahoo_poll_interval_seconds": 60,
        "csv_poll_interval_seconds": 300,
    }
except yaml.YAMLError as e:
    logging.error(f"Error loading config.yaml: {e}. Using default configuration values.")
    config = { # Default values if YAML loading fails
        "db_path": "./data/prices.db",
        "db_ttl_hours": 8,
        "default_resolution": "1m",
        "max_historical": "6 months",
        "symbol_rate_limit_seconds": 1,
        "max_candidate_counts": 3,
        "min_wave_duration_seconds": 60,
        "max_wave_duration_days": 7,
        "elliott_rule_strictness": "moderate",
        "confidence_weights": {
            "rule_compliance": 0.6,
            "amplitude_duration_norm": 0.2,
            "volatility_penalty": 0.1,
            "chaos_metric": 0.1,
        },
        "count_weights": {
            "rule_compliance": 0.7,
            "avg_segment_conf": 0.3,
        },
        "binance_poll_interval_seconds": 60,
        "yahoo_poll_interval_seconds": 60,
        "csv_poll_interval_seconds": 300,
    }

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
# WebSocket connections
websockets: List[WebSocket] = []

# Current active symbol and its data
active_symbol: Optional[str] = None
# Store latest prices for symbol for analysis context and potential real-time updates
current_prices_df: pd.DataFrame = pd.DataFrame()
# Store latest analysis results
latest_analysis_results: Dict[str, Any] = {}

# Data adapter factory
adapter_factory = AdapterFactory(config)
current_adapter: Optional[DataAdapter] = None

# Analysis pipeline components
wave_detector: Optional[WaveDetector] = None

# Database path and TTL from config
DB_PATH = config.get("db_path", "./data/prices.db")
DB_TTL_HOURS = config.get("db_ttl_hours", 8)

# Rate limiting store (in-memory for the backend process)
# Format: {ip_address: last_request_timestamp}
rate_limit_store: Dict[str, float] = {}
SYMBOL_RATE_LIMIT_SECONDS = config.get("symbol_rate_limit_seconds", 1)


# --- Database Operations ---
def get_db_connection():
    """Provides a database connection."""
    conn = sqlite3.connect(DB_PATH)
    # Return rows as dictionary-like objects for easier access
    conn.row_factory = sqlite3.Row 
    return conn

def init_db():
    """Initializes the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
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
    logger.info(f"Database initialized at {DB_PATH}")

def insert_prices(df: pd.DataFrame, symbol: str):
    """Inserts DataFrame of prices into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure timestamp is in ISO format and timezone-aware (UTC)
    if df.index.tz is None:
        df.index = df.index.tz_localize(datetime.timezone.utc)
    else:
        df.index = df.index.tz_convert(datetime.timezone.utc)
        
    df_to_insert = df.copy()
    df_to_insert.reset_index(inplace=True)
    df_to_insert['symbol'] = symbol
    
    # Prepare data for executemany
    # Columns: symbol, timestamp (ISO format), price, volume
    data_to_insert = [
        (row['symbol'], row['timestamp'].isoformat(), row['price'], row['volume'])
        for index, row in df_to_insert.iterrows()
    ]
    
    try:
        cursor.executemany(
            "INSERT OR IGNORE INTO prices (symbol, timestamp, price, volume) VALUES (?, ?, ?, ?)",
            data_to_insert
        )
        conn.commit()
        logger.info(f"Inserted {len(data_to_insert)} rows for {symbol}.")
    except Exception as e:
        logger.error(f"Error inserting prices for {symbol}: {e}")
    finally:
        conn.close()

def fetch_historical_prices_from_db(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    """Fetches historical prices from the DB for a given symbol and date range."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure dates are ISO format for SQL comparison
    start_date_iso = start_date.isoformat()
    end_date_iso = end_date.isoformat()
    
    try:
        cursor.execute(
            "SELECT timestamp, price, volume FROM prices WHERE symbol = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (symbol, start_date_iso, end_date_iso)
        )
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"No historical data found in DB for {symbol} between {start_date} and {end_date}.")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
            
        df = pd.DataFrame(rows)
        # The 'timestamp' column is text, convert it back to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'timestamp': 'timestamp'}, inplace=True) # Keep column name if needed, or just use index
        
        # Ensure output DataFrame matches expected format (index=timestamp, columns=['price', 'volume'])
        # Reorder columns and ensure correct names if necessary (using index as timestamp)
        df = df[['price', 'volume']]
        logger.info(f"Fetched {len(df)} historical price rows from DB for {symbol}.")
        return df

    except Exception as e:
        logger.error(f"Error fetching historical prices from DB for {symbol}: {e}")
        return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    finally:
        conn.close()

def cleanup_old_db_data(ttl_hours: int):
    """Deletes rows older than ttl_hours from the prices table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=ttl_hours)
    cutoff_iso = cutoff_time.isoformat()
    
    try:
        # Use timestamp TEXT comparison for SQLite
        cursor.execute("DELETE FROM prices WHERE timestamp < ?", (cutoff_iso,))
        deleted_count = cursor.rowcount
        conn.commit()
        logger.info(f"Database TTL cleanup: Deleted {deleted_count} rows older than {ttl_hours} hours.")
    except Exception as e:
        logger.error(f"Error during database TTL cleanup: {e}")
    finally:
        conn.close()

# --- Background Worker Tasks ---
def background_db_cleanup_worker():
    """Runs the DB cleanup task periodically."""
    while True:
        cleanup_old_db_data(DB_TTL_HOURS)
        # Sleep for a duration, e.g., half the TTL or a fixed interval
        # Ensure it runs at least once per TTL period.
        sleep_interval = max(DB_TTL_HOURS * 3600 / 2, 3600) # Run cleanup at least twice per TTL, or every hour minimum
        time.sleep(sleep_interval)

async def fetch_and_analyze_data():
    """Fetches data, runs analysis, and pushes updates via WebSocket."""
    global current_adapter, active_symbol, current_prices_df, latest_analysis_results
    
    if not active_symbol or not current_adapter:
        # logger.debug("No active symbol or adapter, skipping fetch/analyze.")
        return

    logger.info(f"Fetching latest data for {active_symbol}...")
    try:
        # Fetch latest data point(s)
        latest_data_df = await current_adapter.fetch_latest_data()
        
        if latest_data_df is not None and not latest_data_df.empty:
            # Append new data to existing DataFrame, maintaining symbol context
            # Ensure timestamp is timezone-aware UTC
            if latest_data_df.index.tz is None:
                 latest_data_df.index = latest_data_df.index.tz_localize(datetime.timezone.utc)
            else:
                 latest_data_df.index = latest_data_df.index.tz_convert(datetime.timezone.utc)

            # Combine with existing data if not already present
            # Filter out any timestamps that are already in current_prices_df to avoid duplicates
            new_data_df = latest_data_df[~latest_data_df.index.isin(current_prices_df.index)]
            
            if not new_data_df.empty:
                current_prices_df = pd.concat([current_prices_df, new_data_df])
                current_prices_df = current_prices_df[~current_prices_df.index.duplicated(keep='last')] # Keep last duplicate if any
                current_prices_df.sort_index(inplace=True)
                
                logger.info(f"Appended {len(new_data_df)} new data points. Total prices: {len(current_prices_df)}")
                
                # --- Save to DB ---
                # Only save new data to DB to avoid redundant writes
                insert_prices(new_data_df, active_symbol)

                # --- Run Analysis Pipeline ---
                logger.info(f"Running analysis for {active_symbol} with {len(current_prices_df)} data points...")
                
                # Resample if necessary to match default resolution or desired interval for analysis
                # For now, assume analysis works on the raw data or a resampled version if needed by WaveDetector
                # WaveDetector needs raw price data and potentially historical context.
                # Ensure the detector is initialized with correct parameters.
                if wave_detector:
                    # Pass the relevant part of current_prices_df for analysis.
                    # WaveDetector will handle its own resampling/smoothing internally based on config.
                    analysis_results = wave_detector.detect_waves(current_prices_df)
                    latest_analysis_results[active_symbol] = analysis_results
                    logger.info(f"Analysis complete for {active_symbol}. Found {len(analysis_results.get('wave_levels', []))} wave levels.")
                    
                    # --- Push Update to WebSockets ---
                    await send_analysis_update(analysis_results)
                else:
                    logger.warning("Wave detector not initialized. Skipping analysis.")
            else:
                # logger.debug("No new data points to append.")
                pass
        else:
            logger.warning(f"Received empty or None data from adapter for {active_symbol}.")

    except Exception as e:
        logger.error(f"Error in fetch_and_analyze_data for {active_symbol}: {e}", exc_info=True)

async def fetch_historical_data_task(symbol: str, adapter: DataAdapter, max_historical_months: int):
    """Fetches historical data from the adapter and populates the DB."""
    global current_prices_df, latest_analysis_results, wave_detector
    
    logger.info(f"Fetching historical data for {symbol}...")
    end_date = datetime.datetime.now(datetime.timezone.utc)
    # Calculate start date based on max_historical_months
    # This needs careful parsing of string like "6 months"
    try:
        # Simple parsing for common string formats
        months_match = int(max_historical_months) if isinstance(max_historical_months, int) else 0
        if isinstance(max_historical_months, str):
            if "month" in max_historical_months:
                try:
                    months_match = int(max_historical_months.split()[0])
                except ValueError:
                    months_match = 6 # Default if parsing fails
            elif "day" in max_historical_months:
                days_match = int(max_historical_months.split()[0])
                start_date = end_date - datetime.timedelta(days=days_match)
                months_match = 0 # Indicate days are used
            else:
                 months_match = 6 # Default
        
        if months_match > 0:
            # Calculate start date by subtracting months, handling year wrap-around
            year = end_date.year
            month = end_date.month
            day = end_date.day
            
            for _ in range(months_match):
                month -= 1
                if month == 0:
                    month = 12
                    year -= 1
            start_date = end_date.replace(year=year, month=month, day=day)
            # If start_date becomes today's date but in a past month, adjust day if it doesn't exist (e.g., March 31st to Feb)
            try:
                start_date = end_date.replace(year=year, month=month, day=day)
            except ValueError: # e.g. trying to set day=31 in February
                start_date = end_date.replace(year=year, month=month, day=1) # Default to first day of month

        else: # days_match is used
            pass # start_date already set

    except Exception as e:
        logger.error(f"Could not parse max_historical '{max_historical_months}': {e}. Defaulting to 6 months.")
        start_date = end_date - datetime.timedelta(days=180) # Fallback to ~6 months

    logger.info(f"Fetching historical data from {start_date.isoformat()} to {end_date.isoformat()}")
    
    historical_df = await adapter.fetch_historical_data(start_date, end_date)
    
    if historical_df is not None and not historical_df.empty:
        # Save fetched data to DB
        insert_prices(historical_df, symbol)
        logger.info(f"Successfully fetched and saved {len(historical_df)} historical data points for {symbol}.")
        
        # Update current_prices_df with historical data
        # Ensure it's timezone-aware UTC
        if historical_df.index.tz is None:
             historical_df.index = historical_df.index.tz_localize(datetime.timezone.utc)
        else:
             historical_df.index = historical_df.index.tz_convert(datetime.timezone.utc)
        
        current_prices_df = historical_df.copy() # Replace with historical data
        logger.info(f"Set current prices to {len(current_prices_df)} historical points.")
        
        # Run initial analysis
        if wave_detector:
            analysis_results = wave_detector.detect_waves(current_prices_df)
            latest_analysis_results[symbol] = analysis_results
            logger.info(f"Initial analysis complete for {symbol}.")
            await send_analysis_update(analysis_results)
    else:
        logger.warning(f"No historical data fetched for {symbol}. Using sample CSV as fallback if available.")
        # Fallback to sample CSV if adapter returns no data and it's not already populated.
        # This logic might need refinement to ensure we don't overwrite existing good data.
        # For now, if adapter fails, we rely on the initial state (empty or whatever was there).
        # If current_prices_df is empty, we might want to load from CSV.
        if current_prices_df.empty and adapter and adapter.interval:
            # Define start_date for CSV loading (same as for historical data)
            start_date = end_date - datetime.timedelta(days=180) # Default to ~6 months
            try:
                csv_path = f"data/sample_{adapter.interval}.csv"
                sample_csv_adapter = SampleCSVAdapter(symbol=symbol, interval=adapter.interval, csv_path=csv_path)
                sample_data = await sample_csv_adapter.fetch_historical_data(start_date, end_date)
                if not sample_data.empty:
                    insert_prices(sample_data, symbol) # Save sample data to DB if loaded
                    current_prices_df = sample_data
                    logger.info(f"Loaded {len(current_prices_df)} points from sample CSV.")
                    if wave_detector:
                        analysis_results = wave_detector.detect_waves(current_prices_df)
                        latest_analysis_results[symbol] = analysis_results
                        await send_analysis_update(analysis_results)
                else:
                    logger.warning(f"Could not load data from sample CSV: {csv_path}")
            except Exception as e:
                logger.error(f"Error loading sample CSV data: {e}")

# Global variable to control the data polling loop
running = True


async def data_polling_loop():
    """Main loop for fetching and analyzing data."""
    global running
    while running:
        if active_symbol and current_adapter:
            await fetch_and_analyze_data()
        else:
            # logger.debug("No active symbol or adapter. Waiting...")
            pass
        
        # Wait for the next poll interval, or a default if not set
        poll_interval = await current_adapter.get_poll_interval() if current_adapter else 60 # Default to 60s if adapter missing
        # Use a loop to check if we should stop early
        for _ in range(int(poll_interval)):
            if not running:
                return
            await asyncio.sleep(1)
        # Wait for the next poll interval, or a default if not set
        poll_interval = await current_adapter.get_poll_interval() if current_adapter else 60 # Default to 60s if adapter missing
        await asyncio.sleep(poll_interval)

# --- WebSocket Management ---
async def send_analysis_update(analysis_results: Dict[str, Any]):
    """Sends analysis results to all connected WebSocket clients."""
    if not websockets:
        return

    frame_schema = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "symbol": active_symbol,
        "prices": [], # Latest prices within a recent window
        "wave_levels": [],
        "candidates": [],
        "predictions": [],
        "technical_indicators": {},
        "ml_predictions": []
    }
    
    # Populate prices - take the last N points, e.g., 500
    if not current_prices_df.empty:
        recent_prices = current_prices_df.tail(500).reset_index()
        frame_schema["prices"] = [
            {"t": row['timestamp'].isoformat(), "p": row['price']}
            for index, row in recent_prices.iterrows()
        ]
    
    # Populate wave levels and candidates from analysis results
    if analysis_results:
        frame_schema["wave_levels"] = analysis_results.get("wave_levels", [])
        frame_schema["candidates"] = analysis_results.get("candidates", [])
        frame_schema["predictions"] = analysis_results.get("predictions", []) # Add predictions if generated
        frame_schema["technical_indicators"] = analysis_results.get("technical_indicators", {}) # Add technical indicators
        frame_schema["ml_predictions"] = analysis_results.get("ml_predictions", []) # Add ML predictions

    message = json.dumps(frame_schema)
    
    # Send to all connected websockets
    disconnected_websockets = []
    for ws in websockets:
        try:
            await ws.send_text(message)
        except WebSocketDisconnect:
            disconnected_websockets.append(ws)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            disconnected_websockets.append(ws)
            
    # Remove disconnected websockets
    for ws in disconnected_websockets:
        websockets.remove(ws)

# --- FastAPI App ---
app = FastAPI(title="Elliott Wave Predictor API")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# --- Rate Limiting Dependency ---
async def rate_limit_dependency(request: Request, rate_limit_seconds: float = Depends(lambda: SYMBOL_RATE_LIMIT_SECONDS)):
    client_ip = request.client.host # Get IP address from request
    current_time = time.time()
    
    last_request_time = rate_limit_store.get(client_ip)
    
    if last_request_time is not None:
        time_since_last_request = current_time - last_request_time
        if time_since_last_request < rate_limit_seconds:
            # Rate limit exceeded
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please try again in {rate_limit_seconds - time_since_last_request:.2f} seconds."
            )
            
    # Update the last request time for this IP
    rate_limit_store[client_ip] = current_time
    return True

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the frontend HTML file."""
    try:
        with open("frontend/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found.</h1><p>Ensure frontend/index.html exists.</p>", status_code=404)

@app.post("/symbol", status_code=200)
async def change_symbol(symbol_data: Dict[str, str], rate_limit_ok: bool = Depends(rate_limit_dependency)):
    """
    Changes the active symbol, triggers historical data fetch and analysis.
    """
    global active_symbol, current_adapter, current_prices_df, latest_analysis_results, wave_detector
    
    new_symbol = symbol_data.get("symbol")
    new_interval = symbol_data.get("interval", config.get("default_resolution", "1m"))
    source_preference = symbol_data.get("source_preference") # e.g., "binance", "yahoo", "csv"

    if not new_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required.")
    
    logger.info(f"Received request to change symbol to: {new_symbol} (Interval: {new_interval}, Source: {source_preference})")

    # --- Update symbol and adapter ---
    active_symbol = new_symbol.upper()
    current_adapter = adapter_factory.get_adapter(active_symbol, interval=new_interval, source_preference=source_preference)
    
    # Clear existing data and analysis results for the new symbol
    current_prices_df = pd.DataFrame()
    latest_analysis_results = {}
    
    # --- Initialize Analysis Pipeline ---
    # Re-initialize WaveDetector if parameters changed (e.g., interval, strictness)
    # It's better to pass config directly to the detector upon initialization.
    wave_detector = WaveDetector(
        config=config,
        rule_engine=ElliottRuleEngine(config), # Pass initialized rule engine
        nms=NMS(), # Pass initialized NMS
        confidence_scorer=ConfidenceScorer(
            weights={
                **config.get('confidence_weights', {}),
                **config.get('count_weights', {})
            }
        )
    )
    logger.info("Wave detector re-initialized for new symbol/settings.")

    # --- Fetch historical data ---
    # This should happen BEFORE starting polling, to have a base for analysis.
    # Use a separate async task so it doesn't block the API response.
    asyncio.create_task(fetch_historical_data_task(
        symbol=active_symbol,
        adapter=current_adapter,
        max_historical_months=config.get("max_historical", "6 months")
    ))

    return {"message": f"Symbol set to {active_symbol}. Fetching historical data and starting analysis."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis updates."""
    await websocket.accept()
    websockets.append(websocket)
    logger.info(f"Client connected to WebSocket. Total clients: {len(websockets)}")
    
    # Send the latest analysis results immediately if available
    if active_symbol and latest_analysis_results.get(active_symbol):
        try:
            await websocket.send_json({
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "symbol": active_symbol,
                "wave_levels": latest_analysis_results[active_symbol].get("wave_levels", []),
                "candidates": latest_analysis_results[active_symbol].get("candidates", []),
                "predictions": latest_analysis_results[active_symbol].get("predictions", []),
                "technical_indicators": latest_analysis_results[active_symbol].get("technical_indicators", {}),
                "ml_predictions": latest_analysis_results[active_symbol].get("ml_predictions", []),
                "prices": [], # Send current prices too
            })
        except Exception as e:
            logger.error(f"Error sending initial message to websocket: {e}")

    try:
        while True:
            data = await websocket.receive_text()
            # We don't expect messages from the client on this endpoint, but can handle them if needed.
            # For now, just acknowledge.
            # logger.debug(f"Received message from WebSocket: {data}")
            pass
    except WebSocketDisconnect:
        websockets.remove(websocket)
        logger.info(f"Client disconnected from WebSocket. Total clients: {len(websockets)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websockets:
            websockets.remove(websocket)

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Initialize DB and start background tasks."""
    logger.info("Starting application...")
    
    # Initialize database
    init_db()
    
    # Start background DB cleanup thread
    db_cleanup_thread = threading.Thread(target=background_db_cleanup_worker, daemon=True)
    db_cleanup_thread.start()
    logger.info("Database cleanup worker started.")
    # Initialize WaveDetector (will be re-initialized on symbol change)
    global wave_detector
    wave_detector = WaveDetector(
        config=config,
        rule_engine=ElliottRuleEngine(config),
        nms=NMS(),
        confidence_scorer=ConfidenceScorer(
            weights={
                **config.get("confidence_weights", {}),
            }
        )
    )
    logger.info("WaveDetector initialized with rule engine, NMS, confidence scorer, technical indicators, and ML predictor")
    """Clean up resources."""
    global running
    logger.info("Shutting down application...")
    running = False
    # Close any open database connections if not using connection pooling
    # (Here, connections are managed per function call, so no explicit close needed globally)
    # Stop background threads if necessary (daemon=True usually handles this)
    logger.info("Application shutdown complete.")
    logger.info("Data polling and analysis loop started.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources."""
    logger.info("Shutting down application...")
    # Close any open database connections if not using connection pooling
    # (Here, connections are managed per function call, so no explicit close needed globally)
    # Stop background threads if necessary (daemon=True usually handles this)
    logger.info("Application shutdown complete.")

# --- Helper for Parsing Max Historical Data ---
def parse_max_historical(max_historical_str: str) -> datetime.timedelta:
    """Parses string like '6 months' into a timedelta object."""
    try:
        parts = max_historical_str.lower().split()
        value = int(parts[0])
        unit = parts[1]
        
        if "month" in unit:
            # Approximate months as days (30.44 days/month on average)
            return datetime.timedelta(days=value * 30.44)
        elif "day" in unit:
            return datetime.timedelta(days=value)
        elif "week" in unit:
            return datetime.timedelta(weeks=value)
        elif "year" in unit:
            return datetime.timedelta(days=value * 365.25)
        else:
            logger.warning(f"Unknown unit for max_historical: {unit}. Defaulting to 30 days.")
            return datetime.timedelta(days=30)
            
    except Exception as e:
        logger.error(f"Could not parse max_historical string '{max_historical_str}': {e}. Defaulting to 6 months (180 days).")
        return datetime.timedelta(days=180)

# --- Function to get config values for adapter factory and detector ---
def get_analysis_config():
    """Returns config relevant for analysis pipeline."""
    return {
        "min_wave_duration_seconds": config.get("min_wave_duration_seconds", 60),
        "max_wave_duration_days": config.get("max_wave_duration_days", 7),
        "elliott_rule_strictness": config.get("elliott_rule_strictness", "moderate"),
        "confidence_weights": config.get("confidence_weights", {}),
        "count_weights": config.get("count_weights", {}),
        "max_historical_timedelta": parse_max_historical(config.get("max_historical", "6 months")),
    }

# --- Main Execution (for running the app) ---
if __name__ == "__main__":
    import uvicorn
    # To run: python backend/main.py
    # Ensure you have dependencies installed: pip install fastapi uvicorn[standard] pandas pyyaml websockets aiohttp sqlite3
    # For adapters: pip install python-binance yfinance
    # For analysis: pip install scipy numpy
    
    # NOTE: Running this directly might require setting up data/ folder and sample CSVs.
    # The frontend files (index.html, app.js, styles.css) should be in frontend/ directory.
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8003)