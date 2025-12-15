import asyncio
import datetime
import logging
import typing
import pandas as pd
import requests
import yfinance as yf
import aiohttp
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration values (can be overridden by config.yaml)
DEFAULT_POLL_INTERVAL_SECONDS = 60 # Default interval for HTTP polling
MAX_HISTORICAL_MONTHS = 6
DEFAULT_RESOLUTION = '1m' # Default to 1-minute data

class DataAdapter(ABC):
    """
    Abstract base class for data adapters.
    Responsible for fetching historical and real-time (polling) price data.
    """
    def __init__(self, symbol: str, interval: str, max_historical_months: int = MAX_HISTORICAL_MONTHS, poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS):
        self.symbol = symbol
        self.interval = interval
        self.max_historical_months = max_historical_months
        self.poll_interval_seconds = poll_interval_seconds
        logger.info(f"Initialized {self.__class__.__name__} for symbol {self.symbol} with interval {self.interval}")

    @abstractmethod
    async def fetch_historical_data(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        """
        Fetches historical data between start_date and end_date.
        Returns a pandas DataFrame with columns: 'timestamp', 'price', 'volume'.
        Timestamp should be timezone-aware (UTC).
        """
        pass

    async def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetches the latest available data point(s).
        This method is intended for polling and should return a small DataFrame.
        The implementation may vary; some APIs provide a 'latest' endpoint, others require fetching a small range.
        """
        # Default implementation: Fetch data for the last `poll_interval_seconds` duration
        # This might not be efficient for all APIs. Specific adapters should override if needed.
        end_date = datetime.datetime.now(datetime.timezone.utc)
        # Calculate start date based on poll_interval_seconds and configured interval resolution.
        # This is a simplification; a more robust approach would consider interval granularity.
        # For example, if interval is '1m', fetch last 5 minutes. If '1h', fetch last hour.
        # For now, we fetch a window that's larger than the poll interval to ensure we catch the latest.
        # We'll aim to fetch at least a few intervals worth of data.
        # A better approach might be to fetch data from last known timestamp + interval.
        # Let's fetch enough data to cover the poll interval and a bit more,
        # considering the smallest common interval is '1m'.
        lookback_minutes = max(5, self.poll_interval_seconds // 60 + 1) # Fetch at least 5 mins, or more if poll interval is long
        start_date = end_date - datetime.timedelta(minutes=lookback_minutes)
        logger.debug(f"Fetching latest data for {self.symbol} from {start_date} to {end_date}")
        return await self.fetch_historical_data(start_date, end_date)


class SampleCSVAdapter(DataAdapter):
    """
    Adapter for sample CSV files.
    Used for testing and demonstration without live API calls.
    """
    def __init__(self, symbol: str, interval: str, csv_path: str, poll_interval_seconds: int = 300):
        super().__init__(symbol, interval, poll_interval_seconds=poll_interval_seconds)
        self.csv_path = csv_path
        self.data = None
        self.last_loaded_timestamp = None
        logger.info(f"Initialized SampleCSVAdapter for symbol {self.symbol} with CSV path {self.csv_path}")

    async def load_data_from_csv(self):
        if self.data is None or (self.last_loaded_timestamp and datetime.datetime.now(datetime.timezone.utc) - self.last_loaded_timestamp > datetime.timedelta(minutes=10)): # Reload if stale
            try:
                df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                # Ensure timezone awareness (UTC)
                if df.index.tz is None:
                    df.index = df.index.tz_localize(datetime.timezone.utc)
                else:
                    df.index = df.index.tz_convert(datetime.timezone.utc)
                # Ensure correct columns, rename if necessary
                df.rename(columns={'price': 'price', 'volume': 'volume'}, inplace=True)
                # Filter for correct interval if possible, though CSVs are usually pre-defined
                # For simplicity, assume CSV is already at the requested interval.
                # If not, resampling would be needed here.
                self.data = df[['price', 'volume']].copy()
                self.last_loaded_timestamp = datetime.datetime.now(datetime.timezone.utc)
                logger.info(f"Successfully loaded data from {self.csv_path}. Found {len(self.data)} rows.")
            except FileNotFoundError:
                logger.error(f"Sample CSV file not found at {self.csv_path}")
                self.data = pd.DataFrame(columns=['price', 'volume']) # Empty DataFrame on error
            except Exception as e:
                logger.error(f"Error loading CSV file {self.csv_path}: {e}")
                self.data = pd.DataFrame(columns=['price', 'volume'])

    async def fetch_historical_data(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        await self.load_data_from_csv()
        if self.data is None or self.data.empty:
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
        # Filter data within the requested date range
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        filtered_data = self.data.loc[mask].copy()
        # Reset index to have timestamp as a column and ensure correct column names
        filtered_data.reset_index(inplace=True)
        filtered_data.rename(columns={'index': 'timestamp'}, inplace=True)
        # Ensure output format matches expected: 'timestamp', 'price', 'volume'
        return filtered_data[['timestamp', 'price', 'volume']]

    async def fetch_latest_data(self) -> pd.DataFrame:
        await self.load_data_from_csv()
        if self.data is None or self.data.empty:
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
        latest_timestamp = self.data.index.max()
        if latest_timestamp:
            # Return the latest row
            latest_row = self.data.loc[[latest_timestamp]].reset_index()
            latest_row.rename(columns={'index': 'timestamp'}, inplace=True)
            return latest_row[['timestamp', 'price', 'volume']]
        else:
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])


class BinanceAdapter(DataAdapter):
    """
    Adapter for Binance API (REST polling).
    Fetches crypto currency data.
    """
    def __init__(self, symbol: str, interval: str = '1m', poll_interval_seconds: int = 60):
        # Binance intervals: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w', '1M'
        # yfinance compatibility might require mapping, but we'll use Binance's native intervals.
        # For this example, we'll assume `interval` is directly compatible with Binance.
        super().__init__(symbol, interval, poll_interval_seconds=poll_interval_seconds)
        self.base_url = "https://api.binance.com"
        # Binance symbol format: BTCUSDT, ETHBTC, etc.
        if not symbol.endswith(('USDT', 'BTC', 'ETH', 'BNB', 'BUSD', 'XRP', 'ADA', 'DOGE', 'SHIB')): # Heuristic for common crypto pairs
             logger.warning(f"Symbol '{symbol}' may not be a standard Binance crypto pair. Expected format like BTCUSDT.")
        self.binance_symbol = symbol.upper() # Binance symbols are uppercase
        logger.info(f"Initialized BinanceAdapter for symbol {self.binance_symbol} with interval {self.interval}")

    async def fetch_historical_data(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        # Binance API uses Unix timestamps in milliseconds
        start_ts_ms = int(start_date.timestamp() * 1000)
        end_ts_ms = int(end_date.timestamp() * 1000)
        # Map user interval to Binance interval
        binance_interval = self.map_interval_to_binance(self.interval)
        if not binance_interval:
            logger.error(f"Unsupported interval for Binance: {self.interval}")
            return pd.DataFrame(columns=["timestamp", "price", "volume"])

        klines_url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": self.binance_symbol,
            "interval": binance_interval,
            "startTime": start_ts_ms,
            "endTime": end_ts_ms,
            "limit": 1000 # Max limit per request
        }

        all_klines = []
        current_start_ts = start_ts_ms

        try:
            async with aiohttp.ClientSession() as session:
                while current_start_ts < end_ts_ms:
                    params["startTime"] = current_start_ts
                    # Ensure endTime is not in the future and not excessively large if current_start_ts is near end_ts_ms
                    params["endTime"] = min(end_ts_ms + self.poll_interval_seconds * 1000, current_start_ts + 1000 * 60 * 60 * 24 * 365) # Limit fetch to a year at once for safety/efficiency

                    async with session.get(klines_url, params=params) as response:
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        klines = await response.json()
                        all_klines.extend(klines)

                        # Set next startTime to the close time of the last kline in the current batch
                        if klines:
                            last_kline_close_time = klines[-1][0]
                            current_start_ts = last_kline_close_time + 1 # Start from the next millisecond after the last close

                        if len(klines) < params.get('limit', 500): # If fewer than limit, assume end of data for this range
                            break
                    await asyncio.sleep(0.1)

            if not all_klines:
                logger.warning(f"No klines found for {self.binance_symbol} in range {start_date} to {end_date}")
                return pd.DataFrame(columns=['timestamp', 'price', 'volume'])

            # Parse klines into DataFrame
            # Each kline is: [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            df = pd.DataFrame(all_klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            # Convert timestamps to datetime objects (milliseconds)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            # We'll use 'close' price for price and 'volume' for volume.
            df['price'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            # Select and reorder columns
            df = df[['timestamp', 'price', 'volume']]
            df.set_index('timestamp', inplace=True) # Use timestamp as index for easier manipulation
            # Ensure index is timezone-aware (UTC)
            if df.index.tz is None:
                df.index = df.index.tz_localize(datetime.timezone.utc)
            else:
                df.index = df.index.tz_convert(datetime.timezone.utc)
            logger.info(f"Fetched {len(df)} klines for {self.binance_symbol} from Binance.")
            return df.reset_index() # Return with timestamp as a column

        except aiohttp.ClientError as e:
            logger.error(f"Binance API request failed for {self.binance_symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
        except Exception as e:
            logger.error(f"Error processing Binance data for {self.binance_symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])

    def map_interval_to_binance(self, interval: str) -> typing.Optional[str]:
        """Maps internal interval string to Binance API interval string."""
        interval_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '1w': '1w', '1M': '1M'
        }
        return interval_map.get(interval.lower())


class MEXCAdapter(DataAdapter):
    """
    Adapter for MEXC API (REST polling).
    Fetches crypto currency data.
    """
    def __init__(self, symbol: str, interval: str = '1m', poll_interval_seconds: int = 60):
        # MEXC intervals: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'
        super().__init__(symbol, interval, poll_interval_seconds=poll_interval_seconds)
        self.base_url = "https://api.mexc.com"
        # MEXC symbol format: BTC_USDT, ETH_USDT, etc.
        if '_' not in symbol and symbol.endswith(('USDT', 'BTC', 'ETH', 'BNB', 'BUSD')):
            # Convert BTCUSDT to BTC_USDT format for MEXC
            for suffix in ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']:
                if symbol.endswith(suffix):
                    self.mexc_symbol = symbol[:-len(suffix)] + '_' + suffix
                    break
        else:
            self.mexc_symbol = symbol.upper() # MEXC symbols are uppercase with underscore
            
        logger.info(f"Initialized MEXCAdapter for symbol {self.mexc_symbol} with interval {self.interval}")

    async def fetch_historical_data(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        # MEXC API uses Unix timestamps in seconds
        start_ts = int(start_date.timestamp())
        # Map user interval to MEXC interval
        mexc_interval = self.map_interval_to_mexc(self.interval)
        if not mexc_interval:
            logger.error(f"Unsupported interval for MEXC: {self.interval}")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])

        klines_url = f"{self.base_url}/api/v1/kline"
        params = {
            'symbol': self.mexc_symbol,
            'interval': mexc_interval,
            'start_time': start_ts,
            'limit': 1000 # Max limit per request
        }

        all_klines = []

        try:
            # For MEXC, we need to paginate through the data
            current_start_ts = start_ts
            
            while current_start_ts < int(end_date.timestamp()):
                params['start_time'] = current_start_ts
                
                response = requests.get(klines_url, params=params)
                response.raise_for_status()

                data = response.json()
                # MEXC returns data in a different format than Binance
                if 'data' in data and data['data']:
                    klines = data['data']
                    all_klines.extend(klines)
                    
                    # Update start time for next iteration
                    if klines:
                        # Last timestamp + interval in seconds
                        interval_seconds = self.get_interval_seconds(mexc_interval)
                        current_start_ts = klines[-1][0] + interval_seconds
                    else:
                        break
                else:
                    break
                # Respect API rate limits
                await asyncio.sleep(0.1)

            if not all_klines:
                logger.warning(f"No data returned from MEXC for {self.mexc_symbol}")
                return pd.DataFrame(columns=['timestamp', 'price', 'volume'])

            # Process klines into DataFrame
            # MEXC kline format: [timestamp, open, close, high, low, volume, amount]
            df_data = []
            for kline in all_klines:
                timestamp = datetime.datetime.fromtimestamp(kline[0], tz=datetime.timezone.utc)
                close_price = float(kline[2])  # Close is at index 2 in MEXC
                volume = float(kline[5])       # Volume is at index 5 in MEXC
                df_data.append({
                    'timestamp': timestamp,
                    'price': close_price,
                    'volume': volume
                })

            df = pd.DataFrame(df_data)
            logger.info(f"Fetched {len(df)} data points from MEXC for {self.mexc_symbol}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"MEXC API request failed for {self.mexc_symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
        except Exception as e:
            logger.error(f"Error processing MEXC data for {self.mexc_symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    def map_interval_to_mexc(self, interval: str) -> typing.Optional[str]:
        """Maps internal interval string to MEXC API interval string."""
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h',
            '1d': '1d', '1w': '1w', '1M': '1M'
        }
        return interval_map.get(interval.lower())
        
    def get_interval_seconds(self, interval: str) -> int:
        """Converts MEXC interval string to seconds."""
        interval_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400,
            '1d': 86400, '1w': 604800, '1M': 2592000
        }
        return interval_map.get(interval, 60)


class YahooFinanceAdapter(DataAdapter):
    """
    Adapter for Yahoo Finance API (via yfinance).
    Fetches stock and index data.
    """
    def __init__(self, symbol: str, interval: str = '1m', poll_interval_seconds: int = 60):
        # yfinance intervals: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        # Note: '1m' interval is only available for the last 7 days and requires a specific `period`.
        # For historical data, '1d' or '1wk' are more reliable.
        super().__init__(symbol, interval, poll_interval_seconds=poll_interval_seconds)
        self.ticker = yf.Ticker(self.symbol)
        logger.info(f"Initialized YahooFinanceAdapter for symbol {self.symbol} with interval {self.interval}")

    async def fetch_historical_data(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        # yfinance expects timezone-naive datetimes or specific timezone handling.
        # Let's convert to UTC and then to naive for yfinance compatibility if needed.
        # However, yfinance's `download` function handles timezone-aware datetimes correctly.
        
        # Ensure start and end dates are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=datetime.timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=datetime.timezone.utc)

        try:
            # yfinance `download` function:
            # If interval is '1m', it requires 'period' and is limited to last 7 days.
            # For longer periods or other intervals, we use start/end dates.
            
            data = None
            if self.interval == '1m':
                # For 1m data, yfinance limits to the last 7 days.
                # Fetching historical data requires `period` to be specified in terms of days.
                # Let's try to fetch up to 7 days. If the requested range is longer, we'll only get the last 7 days.
                # A more complex logic would be needed to handle data older than 7 days for 1m interval.
                # For now, we'll set a generous lookback if the request implies it.
                requested_duration = end_date - start_date
                if requested_duration.days <= 7:
                     data = self.ticker.history(start=start_date, end=end_date, interval=self.interval, auto_adjust=False, back_adjust=False)
                else:
                    # If requested range is > 7 days, we cannot get 1m data for older parts.
                    # We could either warn, or fetch only the last 7 days. Let's fetch the last 7 days.
                    logger.warning(f"Requested 1m data for {self.symbol} older than 7 days. Fetching only the last 7 days.")
                    data = self.ticker.history(period="7d", interval=self.interval, auto_adjust=False, back_adjust=False)
            else:
                # For other intervals, we can fetch longer historical data.
                data = self.ticker.history(start=start_date, end=end_date, interval=self.interval, auto_adjust=False, back_adjust=False)
            
            if data is None or data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {self.symbol}")
                return pd.DataFrame(columns=['timestamp', 'price', 'volume'])

            # yfinance returns a DataFrame with MultiIndex columns if multiple tickers are requested.
            # But for a single ticker, it should have regular columns.
            # Ensure we have the expected columns.
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                logger.error(f"Unexpected data format from Yahoo Finance for {self.symbol}. Columns: {data.columns.tolist()}")
                return pd.DataFrame(columns=['timestamp', 'price', 'volume'])

            # Reset index to make timestamp a column
            data.reset_index(inplace=True)
            
            # Rename columns to match expected format
            data.rename(columns={'Date': 'timestamp', 'Close': 'price', 'Volume': 'volume'}, inplace=True)
            
            # Select only the required columns
            data = data[['timestamp', 'price', 'volume']].copy()
            
            # Ensure timestamp is timezone-aware (UTC)
            if data['timestamp'].dt.tz is None:
                data['timestamp'] = data['timestamp'].dt.tz_localize(datetime.timezone.utc)
            else:
                data['timestamp'] = data['timestamp'].dt.tz_convert(datetime.timezone.utc)
            
            # Convert price and volume to numeric, handling any non-numeric values
            data['price'] = pd.to_numeric(data['price'], errors='coerce')
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
            
            # Drop any rows with NaN values
            data.dropna(inplace=True)
            
            logger.info(f"Fetched {len(data)} data points from Yahoo Finance for {self.symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {self.symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])

    async def fetch_latest_data(self) -> pd.DataFrame:
        """Fetches the latest data point from Yahoo Finance."""
        try:
            # For Yahoo Finance, we'll fetch a small window of recent data
            end_date = datetime.datetime.now(datetime.timezone.utc)
            # For 1m interval, fetch last 15 minutes. For other intervals, fetch last few intervals.
            if self.interval == '1m':
                lookback_delta = datetime.timedelta(minutes=15)
            elif self.interval.endswith('m'): # Other minute intervals
                minutes = int(self.interval[:-1])
                lookback_delta = datetime.timedelta(minutes=minutes * 5) # Fetch 5 intervals worth
            elif self.interval.endswith('h'): # Hour intervals
                hours = int(self.interval[:-1])
                lookback_delta = datetime.timedelta(hours=hours * 3) # Fetch 3 intervals worth
            else: # Day or other intervals
                lookback_delta = datetime.timedelta(hours=2) # Heuristic
            
            start_date = end_date - lookback_delta
            logger.debug(f"Fetching latest data for {self.symbol} from Yahoo Finance (approx. {lookback_delta})")
            return await self.fetch_historical_data(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching latest data from Yahoo Finance for {self.symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])


# --- Adapter Factory and Manager ---

class AdapterFactory:
    """Factory to create appropriate data adapters."""
    def __init__(self, config: dict):
        self.config = config
        # Default values from config.yaml or hardcoded if not present
        self.default_resolution = config.get('default_resolution', DEFAULT_RESOLUTION)
        self.max_historical_months = config.get('max_historical', MAX_HISTORICAL_MONTHS)
        self.symbol_rate_limit_seconds = config.get('symbol_rate_limit_seconds', 1) # This is for /symbol endpoint, not adapter polling
        
        # If specific API keys are in config, they can be used here
        self.api_keys = config.get('api_keys', {})

    def get_adapter(self, symbol: str, interval: typing.Optional[str] = None, source_preference: typing.Optional[str] = None) -> DataAdapter:
        """
        Returns an adapter instance based on symbol, interval, and source preference.
        Source preference can be 'binance', 'yahoo', 'csv', 'mexc'.
        """
        if interval is None:
            interval = self.default_resolution
        
        symbol_upper = symbol.upper()

        # Determine preferred adapter based on symbol and source_preference
        adapter_instance = None
        
        if source_preference == 'mexc':
            logger.info(f"Attempting to use MEXCAdapter for {symbol_upper}")
            adapter_instance = MEXCAdapter(symbol=symbol_upper, interval=interval, poll_interval_seconds=self.config.get('mexc_poll_interval_seconds', DEFAULT_POLL_INTERVAL_SECONDS))
        elif source_preference == 'binance' or (source_preference is None and self._is_crypto_symbol(symbol_upper)):
            logger.info(f"Attempting to use BinanceAdapter for {symbol_upper}")
            # Add API key handling if needed and present in config
            adapter_instance = BinanceAdapter(symbol=symbol_upper, interval=interval, poll_interval_seconds=self.config.get('binance_poll_interval_seconds', DEFAULT_POLL_INTERVAL_SECONDS))
        elif source_preference == 'yahoo' or (source_preference is None and not self._is_crypto_symbol(symbol_upper)):
            logger.info(f"Attempting to use YahooFinanceAdapter for {symbol_upper}")
            # yfinance often works without keys for public data.
            adapter_instance = YahooFinanceAdapter(symbol=symbol_upper, interval=interval, poll_interval_seconds=self.config.get('yahoo_poll_interval_seconds', DEFAULT_POLL_INTERVAL_SECONDS))
        
        # Fallback to CSV adapter if no live adapter can be determined or if explicitly requested
        # The actual CSV fallback will likely be managed by the backend when live adapters fail or are not configured.
        # For now, we'll just instantiate it if specified.
        # In a real scenario, this would be more dynamic.
        if adapter_instance is None and source_preference == 'csv':
            logger.info(f"Using SampleCSVAdapter for {symbol_upper} (fallback/explicit).")
            csv_path = f"data/sample_{interval}.csv" # Assumes interval matches CSV naming
            adapter_instance = SampleCSVAdapter(symbol=symbol_upper, interval=interval, csv_path=csv_path, poll_interval_seconds=self.config.get('csv_poll_interval_seconds', 300)) # Longer interval for CSV

        # If no specific preference or symbol type match, default to Yahoo Finance for general use
        if adapter_instance is None:
            logger.info(f"Defaulting to YahooFinanceAdapter for {symbol_upper}")
            adapter_instance = YahooFinanceAdapter(symbol=symbol_upper, interval=interval, poll_interval_seconds=self.config.get('yahoo_poll_interval_seconds', DEFAULT_POLL_INTERVAL_SECONDS))
            
        return adapter_instance

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Heuristic to determine if a symbol is likely a cryptocurrency."""
        # This is a simple check. A more robust solution might involve a lookup list or API call.
        # Common crypto pairs end with USDT, BTC, ETH, BUSD, etc.
        crypto_suffixes = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD', 'XRP', 'ADA', 'DOGE', 'SHIB']
        return any(symbol.endswith(suffix) for suffix in crypto_suffixes)


# --- Mock adapters for testing and demonstration without real API calls ---
# These would typically be in a separate 'tests.adapters' module

class MockBinanceAdapter(BinanceAdapter):
    async def fetch_historical_data(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        logger.info(f"MockBinanceAdapter: Fetching mock historical data for {self.symbol}")
        # Simulate data
        ts = pd.date_range(start=start_date, end=end_date, freq=self.interval, tz=datetime.timezone.utc)
        if len(ts) == 0:
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
        prices = [p + (p*0.001 * (i % 10 - 5)) for i, p in enumerate([70000 + x*10 for x in range(len(ts))])] # Simulate price movement
        volumes = [1000 + i*50 for i in range(len(ts))]
        df = pd.DataFrame({'timestamp': ts, 'price': prices, 'volume': volumes})
        return df

    async def fetch_latest_data(self) -> pd.DataFrame:
        logger.info(f"MockBinanceAdapter: Fetching mock latest data for {self.symbol}")
        # Simulate latest data point
        now = datetime.datetime.now(datetime.timezone.utc)
        # Adjust timestamp to the expected interval
        interval_dt = datetime.timedelta(minutes=1) # Assuming 1m interval for simulation
        latest_ts = (now - (now - datetime.datetime(2000,1,1,tzinfo=datetime.timezone.utc)) % interval_dt)
        price = 70500.0 + (latest_ts.minute * 10) # Simple price simulation
        volume = 1500 + latest_ts.minute * 50
        df = pd.DataFrame([{'timestamp': latest_ts, 'price': price, 'volume': volume}])
        return df


class MockYahooFinanceAdapter(YahooFinanceAdapter):
    async def fetch_historical_data(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        logger.info(f"MockYahooFinanceAdapter: Fetching mock historical data for {self.symbol}")
        # Simulate data
        ts = pd.date_range(start=start_date, end=end_date, freq=self.interval, tz=datetime.timezone.utc)
        if len(ts) == 0:
            return pd.DataFrame(columns=['timestamp', 'price', 'volume'])
        prices = [p + (p*0.0005 * (i % 5 - 2)) for i, p in enumerate([150 + x*2 for x in range(len(ts))])] # Simulate price movement
        volumes = [10000 + i*100 for i in range(len(ts))]
        df = pd.DataFrame({'timestamp': ts, 'price': prices, 'volume': volumes})
        return df

    async def fetch_latest_data(self) -> pd.DataFrame:
        logger.info(f"MockYahooFinanceAdapter: Fetching mock latest data for {self.symbol}")
        # Simulate latest data point
        now = datetime.datetime.now(datetime.timezone.utc)
        interval_dt = datetime.timedelta(hours=1) # Assuming 1h interval for simulation
        latest_ts = (now - (now - datetime.datetime(2000,1,1,tzinfo=datetime.timezone.utc)) % interval_dt)
        price = 155.0 + (latest_ts.hour * 5)
        volume = 12000 + latest_ts.hour * 100
        df = pd.DataFrame([{'timestamp': latest_ts, 'price': price, 'volume': volume}])
        return df


class MockCSVAdapter(SampleCSVAdapter):
    # For testing, ensure the CSV path is handled.
    # In a real scenario, we'd ensure the sample CSVs exist.
    pass # Inherits logic, no need to override for basic mock behavior


# Helper function to get an adapter instance
def get_data_adapter(symbol: str, interval: typing.Optional[str] = None, source_preference: typing.Optional[str] = None, config: dict = None) -> DataAdapter:
    """
    Convenience function to get a data adapter instance.
    This would typically be called by the backend.
    """
    if config is None:
        config = {} # Default empty config
    
    factory = AdapterFactory(config)
    return factory.get_adapter(symbol, interval, source_preference)