"""High-performance parallel data fetching engine for Nifty 500."""

import logging
import sqlite3
import time
import json
import os
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import pandas as pd
import yfinance as yf
import requests
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PostgreSQL support (optional - for Streamlit Cloud persistence)
try:
    import psycopg2
    from psycopg2.extras import execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.info("PostgreSQL support not available (psycopg2 not installed). Using SQLite only.")


class FastDataEngine:
    """High-performance data engine with parallel fetching and intelligent caching."""

    def __init__(self, db_path: str | None = None, max_workers: int = config.MAX_WORKERS):
        """
        Initialize fast data engine.

        Args:
            db_path: Path to SQLite database (ignored if PostgreSQL is configured)
            max_workers: Maximum threads for parallel fetching
        """
        self.db_path = db_path or str(config.DB_PATH)
        self.max_workers = max_workers
        self._company_names_cache: Dict[str, str] = {}  # In-memory cache (backup)
        
        # Check for PostgreSQL connection from Streamlit secrets
        self.use_postgres = False
        self.postgres_url = None
        
        if POSTGRES_AVAILABLE:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and st.secrets:
                    db_secrets = st.secrets.get("database", {})
                    self.postgres_url = db_secrets.get("postgres_url") or db_secrets.get("url")
                    if self.postgres_url:
                        self.use_postgres = True
                        logger.info("PostgreSQL connection detected from Streamlit secrets. Using PostgreSQL for persistent storage.")
            except Exception as e:
                logger.debug(f"Could not access Streamlit secrets: {e}. Using SQLite.")
        
        self._init_database()

    def _get_connection(self):
        """
        Get database connection (SQLite or PostgreSQL).
        
        Returns:
            Database connection object
        """
        if self.use_postgres and self.postgres_url:
            if not POSTGRES_AVAILABLE:
                raise RuntimeError("PostgreSQL URL configured but psycopg2 not available")
            conn = psycopg2.connect(self.postgres_url)
            conn.autocommit = False  # Use transactions
            return conn
        else:
            # SQLite connection
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # Wait up to 30s for lock
                check_same_thread=False  # Allow multi-threaded access
            )
            # Enable WAL mode for better concurrency (SQLite only)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                # Enable foreign keys (for future use)
                conn.execute("PRAGMA foreign_keys=ON")
            except Exception:
                pass  # Ignore if PRAGMA fails
            return conn

    @contextmanager
    def _db_connection(self):
        """
        Context manager for database connections with automatic cleanup.
        
        Usage:
            with self._db_connection() as conn:
                # ... operations ...
        """
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _get_placeholder(self) -> str:
        """Get SQL placeholder for current database type."""
        return "%s" if self.use_postgres else "?"
    
    def _get_upsert_sql(self, table: str, columns: List[str], conflict_cols: List[str]) -> str:
        """
        Get UPSERT SQL statement compatible with both SQLite and PostgreSQL.
        
        Args:
            table: Table name
            columns: List of column names
            conflict_cols: List of columns that define the conflict (for UNIQUE constraint)
            
        Returns:
            SQL UPSERT statement
        """
        placeholder = self._get_placeholder()
        cols_str = ", ".join(columns)
        values_str = ", ".join([placeholder] * len(columns))
        conflict_str = ", ".join(conflict_cols)
        
        if self.use_postgres:
            # PostgreSQL: INSERT ... ON CONFLICT ... DO UPDATE
            update_cols = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col not in conflict_cols])
            return f"""
                INSERT INTO {table} ({cols_str})
                VALUES ({values_str})
                ON CONFLICT ({conflict_str}) DO UPDATE SET {update_cols}
            """
        else:
            # SQLite: INSERT OR REPLACE
            return f"""
                INSERT OR REPLACE INTO {table} ({cols_str})
                VALUES ({values_str})
            """
    
    def _init_database(self) -> None:
        """Initialize database tables with integrity constraints."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            
            # OHLCV Data table with enhanced constraints
            if self.use_postgres:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv_data (
                        ticker TEXT NOT NULL,
                        date DATE NOT NULL,
                        open DOUBLE PRECISION NOT NULL CHECK(open > 0),
                        high DOUBLE PRECISION NOT NULL CHECK(high > 0),
                        low DOUBLE PRECISION NOT NULL CHECK(low > 0),
                        close DOUBLE PRECISION NOT NULL CHECK(close > 0),
                        volume INTEGER NOT NULL CHECK(volume >= 0),
                        last_updated TIMESTAMP NOT NULL,
                        PRIMARY KEY (ticker, date),
                        CHECK(high >= low),
                        CHECK(high >= open),
                        CHECK(high >= close),
                        CHECK(low <= open),
                        CHECK(low <= close)
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv_data (
                        ticker TEXT NOT NULL,
                        date DATE NOT NULL,
                        open REAL NOT NULL CHECK(open > 0),
                        high REAL NOT NULL CHECK(high > 0),
                        low REAL NOT NULL CHECK(low > 0),
                        close REAL NOT NULL CHECK(close > 0),
                        volume INTEGER NOT NULL CHECK(volume >= 0),
                        last_updated TIMESTAMP NOT NULL,
                        PRIMARY KEY (ticker, date),
                        CHECK(high >= low),
                        CHECK(high >= open),
                        CHECK(high >= close),
                        CHECK(low <= open),
                        CHECK(low <= close)
                    )
                """)
            
            # Company Names table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS company_names (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    last_updated TIMESTAMP NOT NULL
                )
            """)
            
            # Market Intelligence table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_intelligence (
                    symbol TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    last_updated TIMESTAMP NOT NULL
                )
            """)
            
            # Strategy Results table - stores computed trade signals
            if self.use_postgres:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_results (
                        id BIGSERIAL PRIMARY KEY,
                        ticker TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        signals_json TEXT NOT NULL,
                        data_hash TEXT NOT NULL,
                        last_updated TIMESTAMP NOT NULL,
                        UNIQUE(ticker, regime, data_hash)
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        signals_json TEXT NOT NULL,
                        data_hash TEXT NOT NULL,
                        last_updated TIMESTAMP NOT NULL,
                        UNIQUE(ticker, regime, data_hash)
                    )
                """)
            
            # Indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date 
                ON ohlcv_data(ticker, date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_last_updated 
                ON ohlcv_data(ticker, last_updated)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_company_names_updated 
                ON company_names(last_updated)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_intel_updated 
                ON market_intelligence(last_updated)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategy_results_ticker_regime 
                ON strategy_results(ticker, regime)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategy_results_updated 
                ON strategy_results(last_updated)
            """)
            
            conn.commit()

    def _load_company_name(self, ticker: str) -> Optional[str]:
        """
        Load company name from database cache.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company name if found in cache, None otherwise
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                placeholder = self._get_placeholder()
                cursor.execute(f"""
                    SELECT company_name FROM company_names
                    WHERE ticker = {placeholder}
                """, (ticker,))
                result = cursor.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            logger.debug(f"Error loading company name for {ticker}: {e}")
        return None

    def _save_company_name(self, ticker: str, company_name: str) -> None:
        """
        Save company name to database cache with transaction.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name
        """
        try:
            with self._db_connection() as conn:
                try:
                    if not self.use_postgres:
                        if not self.use_postgres:
                        conn.execute("BEGIN TRANSACTION")
                    cursor = conn.cursor()
                    placeholder = self._get_placeholder()
                    if self.use_postgres:
                        cursor.execute(f"""
                            INSERT INTO company_names
                            (ticker, company_name, last_updated)
                            VALUES ({placeholder}, {placeholder}, {placeholder})
                            ON CONFLICT (ticker) DO UPDATE SET
                            company_name = EXCLUDED.company_name,
                            last_updated = EXCLUDED.last_updated
                        """, (ticker, company_name, datetime.now().isoformat()))
                    else:
                        cursor.execute(f"""
                            INSERT OR REPLACE INTO company_names
                            (ticker, company_name, last_updated)
                            VALUES ({placeholder}, {placeholder}, {placeholder})
                        """, (ticker, company_name, datetime.now().isoformat()))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to save company name for {ticker}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error saving company name for {ticker}: {e}")

    def get_company_name(self, ticker: str) -> str:
        """
        Get company name for a ticker, using cache first, then yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company name or ticker if unavailable
        """
        # Try cache first
        cached_name = self._load_company_name(ticker)
        if cached_name:
            return cached_name
        
        # Fetch from yfinance if not in cache
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Check if info is None
            if info is None:
                raise ValueError("stock.info returned None")
            company_name = info.get("longName") or info.get("shortName") or info.get("name", "")
            if company_name:
                # Save to cache
                self._save_company_name(ticker, company_name)
                return company_name
        except Exception as e:
            logger.debug(f"Could not fetch company name for {ticker}: {e}")
        
        # Fallback: return ticker without .NS suffix
        fallback_name = ticker.replace(".NS", "").replace("^", "")
        # Save fallback to cache to avoid repeated API calls
        self._save_company_name(ticker, fallback_name)
        return fallback_name

    def _fetch_single_ticker(self, ticker: str, period: str = "1y") -> tuple[str, pd.DataFrame]:
        """
        Fetch data for a single ticker with raw prices (auto_adjust=False).
        
        CRITICAL: Uses auto_adjust=False to match broker prices (Dhan, etc.)
        Raw Close price without dividend adjustments.
        
        Includes rate limiting to prevent IP blocking.

        Args:
            ticker: Stock ticker symbol
            period: Data period

        Returns:
            Tuple of (ticker, DataFrame)
        """
        try:
            # Rate limiting: Small delay to prevent IP blocking (0.1-0.2 seconds per request)
            time.sleep(0.15)
            
            stock = yf.Ticker(ticker)
            # CRITICAL: auto_adjust=False to get raw prices matching broker standards
            df = stock.history(period=period, auto_adjust=False)
            
            # Check if df is None or empty
            if df is None or df.empty:
                logger.warning(f"No data for {ticker}")
                return ticker, pd.DataFrame()
            
            # Check if required columns exist
            if "Date" not in df.columns and df.index.name != "Date":
                # Try to reset index if Date is in index
                if df.index.name is None or df.index.name == "Date":
                    df = df.reset_index()
                else:
                    logger.warning(f"Unexpected data format for {ticker}")
                    return ticker, pd.DataFrame()
            
            df = df.reset_index()
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            
            # Check if date column exists
            if "date" not in df.columns:
                logger.warning(f"No date column for {ticker}")
                return ticker, pd.DataFrame()
            
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["ticker"] = ticker
            df["last_updated"] = datetime.now()
            
            df = df.set_index("date")
            return ticker, df[["open", "high", "low", "close", "volume", "ticker", "last_updated"]]
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return ticker, pd.DataFrame()

    def _validate_ohlcv_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Validate OHLCV data before writing to database.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns for {ticker}")
            return False
        
        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            logger.error(f"Negative prices detected for {ticker}")
            return False
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            logger.error(f"Invalid high/low prices for {ticker}")
            return False
        
        # Check high >= open and high >= close
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            logger.error(f"Invalid high price (must be >= open and close) for {ticker}")
            return False
        
        # Check low <= open and low <= close
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            logger.error(f"Invalid low price (must be <= open and close) for {ticker}")
            return False
        
        # Check volume >= 0
        if (df['volume'] < 0).any():
            logger.error(f"Negative volume detected for {ticker}")
            return False
        
        return True

    def _get_cache_date_range(self, ticker: str) -> Optional[Tuple[date, date]]:
        """
        Get date range of cached data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (min_date, max_date) if cache exists, None otherwise
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                placeholder = self._get_placeholder()
                cursor.execute(f"""
                    SELECT MIN(date) as min_date, MAX(date) as max_date
                    FROM ohlcv_data
                    WHERE ticker = {placeholder}
                """, (ticker,))
                result = cursor.fetchone()
                if result and result[0]:
                    min_date = result[0] if isinstance(result[0], date) else datetime.fromisoformat(str(result[0])).date()
                    max_date = result[1] if isinstance(result[1], date) else datetime.fromisoformat(str(result[1])).date()
                    return (min_date, max_date)
        except Exception as e:
            logger.debug(f"Error getting cache date range for {ticker}: {e}")
        return None

    def _is_cache_valid(self, ticker: str) -> bool:
        """
        Check if cached data is still valid (< 24 hours old).

        Args:
            ticker: Stock ticker

        Returns:
            True if cache is valid, False otherwise
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                placeholder = self._get_placeholder()
                cursor.execute(f"""
                    SELECT MAX(last_updated) 
                    FROM ohlcv_data 
                    WHERE ticker = {placeholder}
                """, (ticker,))
                
                result = cursor.fetchone()
                
                if result and result[0]:
                    last_updated = datetime.fromisoformat(result[0])
                    age = datetime.now() - last_updated
                    return age < timedelta(hours=config.CACHE_EXPIRY_HOURS)
        except Exception as e:
            logger.debug(f"Error checking cache validity for {ticker}: {e}")
        
        return False

    def _upsert_ohlcv_data(self, conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
        """
        Upsert OHLCV data (insert new, update existing) with transaction.
        
        Args:
            conn: Database connection
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data
        """
        if df.empty:
            return
        
        # Validate data before writing
        if not self._validate_ohlcv_data(df, ticker):
            logger.error(f"Skipping invalid data for {ticker}")
            return
        
        cursor = conn.cursor()
        df_cache = df.copy()
        
        # Reset index to get date as a column if it's in the index
        if df_cache.index.name == 'date' or (isinstance(df_cache.index[0], date) if len(df_cache) > 0 else False):
            df_cache = df_cache.reset_index()
        
        # Prepare data for insertion
        current_time = datetime.now().isoformat()
        
        for _, row in df_cache.iterrows():
            try:
                # Get date from row (either from 'date' column or index)
                if 'date' in row:
                    row_date = row['date']
                elif 'Date' in row:
                    row_date = row['Date']
                else:
                    # Date is in index, use index value
                    row_date = row.name if hasattr(row, 'name') else None
                    if row_date is None:
                        logger.error(f"Could not find date for row in {ticker}")
                        continue
                
                # Convert date to string if needed
                if isinstance(row_date, pd.Timestamp):
                    row_date = row_date.date()
                elif isinstance(row_date, datetime):
                    row_date = row_date.date()
                elif not isinstance(row_date, date):
                    row_date = pd.to_datetime(row_date).date()
                
                placeholder = self._get_placeholder()
                if self.use_postgres:
                    cursor.execute(f"""
                        INSERT INTO ohlcv_data 
                        (ticker, date, open, high, low, close, volume, last_updated)
                        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                        ON CONFLICT (ticker, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        last_updated = EXCLUDED.last_updated
                    """, (
                        ticker,
                        str(row_date),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume']),
                        current_time
                    ))
                else:
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO ohlcv_data 
                        (ticker, date, open, high, low, close, volume, last_updated)
                        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                    """, (
                        ticker,
                        str(row_date),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume']),
                        current_time
                    ))
            except Exception as e:
                logger.error(f"Error upserting row for {ticker}: {e}")
                continue
        
        logger.debug(f"Upserted {len(df_cache)} rows for {ticker}")

    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save DataFrame to cache using UPSERT with transaction.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data
        """
        if df.empty:
            return
        
        try:
            with self._db_connection() as conn:
                try:
                    if not self.use_postgres:
                        conn.execute("BEGIN TRANSACTION")
                    self._upsert_ohlcv_data(conn, ticker, df)
                    conn.commit()
                    logger.debug(f"Cached {len(df)} rows for {ticker}")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to save cache for {ticker}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error saving cache for {ticker}: {e}")

    def _load_from_cache(self, ticker: str, period: str = "1y") -> pd.DataFrame | None:
        """Load data from cache."""
        try:
            with self._db_connection() as conn:
                period_days = self._period_to_days(period)
                cutoff_date = datetime.now().date() - timedelta(days=period_days)
                
                placeholder = self._get_placeholder()
                query = f"""
                    SELECT date, open, high, low, close, volume
                    FROM ohlcv_data
                    WHERE ticker = {placeholder} AND date >= {placeholder}
                    ORDER BY date ASC
                """
                
                df = pd.read_sql(
                    query,
                    conn,
                    params=(ticker, cutoff_date),
                    parse_dates=["date"],
                )
                
                if df.empty:
                    return None
                
                df = df.set_index("date")
                # Ensure index is date objects (not Timestamp) for consistency with _fetch_single_ticker
                # Convert DatetimeIndex to list of date objects explicitly
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.Index([d.date() if isinstance(d, pd.Timestamp) else d for d in df.index], dtype=object)
                elif len(df) > 0:
                    # Convert any other type to date objects
                    df.index = pd.Index([d if isinstance(d, date) else pd.Timestamp(d).date() for d in df.index], dtype=object)
                
                return df
                
        except Exception as e:
            logger.warning(f"Error loading from cache for {ticker}: {e}")
            return None

    def _fetch_incremental(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch incremental data (only missing days) for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (used only if no cache exists)
            
        Returns:
            DataFrame with incremental data
        """
        date_range = self._get_cache_date_range(ticker)
        if date_range:
            # Cache exists - fetch last 5 days to ensure we have latest data
            # yfinance will return only available dates
            fetch_period = "5d"
            logger.debug(f"Incremental fetch for {ticker}: fetching last 5 days")
        else:
            # No cache, fetch full period
            fetch_period = period
            logger.debug(f"Full fetch for {ticker}: no cache exists")
        
        _, df = self._fetch_single_ticker(ticker, fetch_period)
        return df

    def _merge_data(self, cached_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge cached data with new incremental data.
        
        Args:
            cached_df: Existing cached DataFrame
            new_df: New DataFrame from incremental fetch
            
        Returns:
            Merged DataFrame with duplicates removed (new data takes precedence)
        """
        if cached_df.empty:
            return new_df
        if new_df.empty:
            return cached_df
        
        # Ensure both DataFrames have consistent index types (date objects)
        # Convert cached index to date if it's not already
        if len(cached_df) > 0:
            if isinstance(cached_df.index, pd.DatetimeIndex):
                # Convert DatetimeIndex to list of date objects
                cached_df.index = [d.date() if isinstance(d, pd.Timestamp) else d for d in cached_df.index]
            elif len(cached_df) > 0 and not isinstance(cached_df.index[0], date):
                # Convert other types to date
                cached_df.index = pd.to_datetime(cached_df.index).date
        
        # Convert new_df index to date if needed (it should already be date from _fetch_single_ticker)
        if len(new_df) > 0:
            if isinstance(new_df.index, pd.DatetimeIndex):
                # Convert DatetimeIndex to list of date objects
                new_df.index = [d.date() if isinstance(d, pd.Timestamp) else d for d in new_df.index]
            elif len(new_df) > 0 and not isinstance(new_df.index[0], date):
                # Convert other types to date
                new_df.index = pd.to_datetime(new_df.index).date
        
        # Ensure both indices are now date objects (create new index to ensure consistency)
        if len(cached_df) > 0:
            cached_df.index = pd.Index([d if isinstance(d, date) else pd.Timestamp(d).date() for d in cached_df.index], dtype=object)
        if len(new_df) > 0:
            new_df.index = pd.Index([d if isinstance(d, date) else pd.Timestamp(d).date() for d in new_df.index], dtype=object)
        
        # Combine and remove duplicates (keep last = new data)
        combined = pd.concat([cached_df, new_df])
        # Remove duplicates by date index, keeping last (newest)
        merged = combined[~combined.index.duplicated(keep='last')]
        # Sort by date - convert to Timestamp for sorting, then back to date
        # This avoids the comparison error
        merged_index_as_ts = pd.to_datetime([d if isinstance(d, date) else pd.Timestamp(d).date() for d in merged.index])
        merged_sorted = merged.copy()
        merged_sorted.index = merged_index_as_ts
        merged_sorted = merged_sorted.sort_index()
        # Convert back to date objects (convert each Timestamp to date)
        merged_sorted.index = pd.Index([d.date() if isinstance(d, pd.Timestamp) else d for d in merged_sorted.index], dtype=object)
        
        return merged_sorted

    def _period_to_days(self, period: str) -> int:
        """Convert period string to days."""
        period_map: Dict[str, int] = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
        }
        return period_map.get(period, 365)

    def fetch_single(self, ticker: str, period: str = "1y", use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch data for a single ticker with caching, incremental updates, and weekend guardrails.

        Args:
            ticker: Stock ticker
            period: Data period
            use_cache: Whether to use cache
            force_refresh: If True, ignore cache and fetch fresh from yfinance

        Returns:
            DataFrame with OHLCV data (ensures last row is valid trading day)
        """
        # If force_refresh is False, try to use cache first (even if expired)
        if use_cache and not force_refresh:
            cached = self._load_from_cache(ticker, period)
            if cached is not None and len(cached) > 0:
                # Check if cache needs incremental update (older than 1 day)
                date_range = self._get_cache_date_range(ticker)
                if date_range:
                    max_date = date_range[1]
                    days_since_update = (datetime.now().date() - max_date).days
                    if days_since_update > 0:
                        # Cache exists but may be stale - do incremental update
                        logger.info(f"Incremental update for {ticker} (cache is {days_since_update} days old)")
                        new_df = self._fetch_incremental(ticker, period)
                        if not new_df.empty:
                            df = self._merge_data(cached, new_df)
                            self._save_to_cache(ticker, df)
                        else:
                            df = cached
                    else:
                        # Cache is up to date
                        logger.debug(f"Using cache for {ticker} (up to date)")
                        df = cached
                else:
                    df = cached
            else:
                # Cache miss or empty - fetch fresh
                logger.info(f"Cache miss for {ticker}, fetching fresh data")
                _, df = self._fetch_single_ticker(ticker, period)
                if not df.empty and use_cache:
                    self._save_to_cache(ticker, df)
        else:
            # force_refresh=True or use_cache=False - always fetch fresh
            if force_refresh:
                logger.info(f"Force refresh: Fetching fresh data for {ticker}")
            else:
                logger.info(f"Fetching fresh data for {ticker} (cache disabled)")
            _, df = self._fetch_single_ticker(ticker, period)
            if not df.empty and use_cache:
                self._save_to_cache(ticker, df)
        
        # Weekend Guardrail: Ensure last row is a valid trading day
        if not df.empty:
            df = self._apply_weekend_guardrail(df)
        
        # Ensure index is date objects (not Timestamp) for consistency
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.Index([d.date() if isinstance(d, pd.Timestamp) else d for d in df.index], dtype=object)
        elif not df.empty and len(df) > 0 and not isinstance(df.index[0], date):
            df.index = pd.Index([d if isinstance(d, date) else pd.Timestamp(d).date() for d in df.index], dtype=object)
        
        return df[["open", "high", "low", "close", "volume"]] if not df.empty else pd.DataFrame()

    def _apply_weekend_guardrail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weekend guardrail: If today is Saturday/Sunday, use last Friday's data.

        Args:
            df: DataFrame with date index

        Returns:
            DataFrame with valid last trading day
        """
        if df.empty:
            return df
        
        # Get current day of week (0=Monday, 6=Sunday)
        today = datetime.now()
        day_of_week = today.weekday()
        
        # If Saturday (5) or Sunday (6), find last Friday
        if day_of_week >= 5:  # Saturday or Sunday
            # Find last Friday (4)
            days_back = (day_of_week - 4) % 7
            if days_back == 0:
                days_back = 7  # If Sunday, go back 7 days to previous Friday
            
            target_date = today - timedelta(days=days_back)
            
            # Filter to only include data up to target_date
            if df.index.dtype != 'datetime64[ns]':
                df.index = pd.to_datetime(df.index)
            
            df = df.loc[df.index <= target_date]
            logger.debug(f"Weekend guardrail: Using data up to {target_date.date()}")
        
        return df

    def fetch_batch_data(
        self,
        tickers: List[str],
        period: str = "1y",
        use_cache: bool = True,
        force_refresh: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers in parallel using ThreadPoolExecutor.

        Args:
            tickers: List of ticker symbols
            period: Data period
            use_cache: Whether to use cache
            force_refresh: If True, ignore cache and fetch fresh from yfinance
            progress_callback: Optional callback function(completed, total) for progress

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results: Dict[str, pd.DataFrame] = {}
        total = len(tickers)
        completed = 0
        
        # Separate tickers into cached and fresh
        cached_tickers: List[str] = []
        fresh_tickers: List[str] = []
        
        # If force_refresh is False, try to use cache first (even if expired)
        if use_cache and not force_refresh:
            for ticker in tickers:
                cached = self._load_from_cache(ticker, period)
                if cached is not None and len(cached) > 0:
                    results[ticker] = cached
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                else:
                    fresh_tickers.append(ticker)
        else:
            # force_refresh=True or use_cache=False - fetch all fresh
            fresh_tickers = tickers
        
        # Fetch fresh data in parallel with rate limiting
        if fresh_tickers:
            # Limit concurrent requests to prevent overwhelming the API
            # With 0.15s delay per request, 20 workers = ~3 requests/second (safe rate)
            effective_workers = min(self.max_workers, 20)  # Cap at 20 to prevent IP blocking
            
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._fetch_single_ticker, ticker, period): ticker
                    for ticker in fresh_tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        ticker_name, df = future.result()
                        if not df.empty:
                            # Apply weekend guardrail
                            df = self._apply_weekend_guardrail(df)
                            
                            # If incremental update, merge with existing cache
                            if use_cache and not force_refresh:
                                cached = self._load_from_cache(ticker_name, period)
                                if cached is not None and len(cached) > 0:
                                    df = self._merge_data(cached, df)
                            
                            results[ticker_name] = df[["open", "high", "low", "close", "volume"]]
                            if use_cache:
                                self._save_to_cache(ticker_name, df)
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {e}")
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
        
        logger.info(f"Fetched data for {len(results)}/{total} tickers")
        return results

    def filter_dead_stocks(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        min_adx: float = 20.0,
        min_volume: int = 100_000,
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter out dead stocks (low ADX or low volume).

        Args:
            ticker_data: Dictionary mapping ticker to DataFrame
            min_adx: Minimum ADX value (default: 20)
            min_volume: Minimum average volume (default: 100k)

        Returns:
            Filtered dictionary with only active stocks
        """
        import pandas_ta as ta
        
        filtered = {}
        for ticker, df in ticker_data.items():
            if df.empty or len(df) < 20:
                continue
            
            # Check volume
            avg_volume = df["volume"].tail(20).mean()
            if avg_volume < min_volume:
                logger.debug(f"Filtered {ticker}: Low volume ({avg_volume:.0f} < {min_volume})")
                continue
            
            # Check ADX
            try:
                adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
                if isinstance(adx_result, pd.DataFrame):
                    adx_col = "ADX_14"
                    if adx_col not in adx_result.columns:
                        # Try to find ADX column
                        for col in adx_result.columns:
                            if "ADX" in col:
                                adx_col = col
                                break
                    if adx_col in adx_result.columns:
                        current_adx = adx_result[adx_col].iloc[-1]
                        if pd.isna(current_adx) or current_adx < min_adx:
                            logger.debug(f"Filtered {ticker}: Low ADX ({current_adx:.1f} < {min_adx})")
                            continue
                elif isinstance(adx_result, pd.Series):
                    current_adx = adx_result.iloc[-1]
                    if pd.isna(current_adx) or current_adx < min_adx:
                        logger.debug(f"Filtered {ticker}: Low ADX ({current_adx:.1f} < {min_adx})")
                        continue
            except Exception as e:
                logger.debug(f"Error calculating ADX for {ticker}: {e}")
                continue
            
            filtered[ticker] = df
        
        logger.info(f"Filtered {len(filtered)}/{len(ticker_data)} stocks (ADX >= {min_adx}, Volume >= {min_volume:,})")
        return filtered

    def check_market_regime(self, index_symbol: str = config.MARKET_INDEX) -> bool:
        """
        Check if market is in bullish regime (above 200 EMA).

        Args:
            index_symbol: Market index symbol

        Returns:
            True if market is above 200 EMA (bullish), False otherwise
        """
        df = self.fetch_single(index_symbol, period="1y", use_cache=True)
        if df.empty or len(df) < 200:
            logger.warning(f"Insufficient data for regime check: {index_symbol}")
            return False
        
        import pandas_ta as ta
        
        df["ema_200"] = ta.ema(df["close"], length=200)
        current_price = df["close"].iloc[-1]
        ema_200 = df["ema_200"].iloc[-1]
        
        is_bullish = current_price > ema_200
        logger.info(
            f"Market regime: {index_symbol} - Price: {current_price:.2f}, "
            f"200 EMA: {ema_200:.2f}, Bullish: {is_bullish}"
        )
        return is_bullish

    def get_nifty500_tickers(self, use_cache: bool = True) -> List[str]:
        """
        Dynamically fetch Nifty 500 ticker list from official NSE website.
        Automatically handles rebalancing by fetching live data.
        Appends .NS suffix for yfinance compatibility.
        
        NOTE: If NSE website is unavailable, falls back to default list from config.
        You can manually update the ticker list by placing a CSV file at:
        data/nifty500_tickers_cache.csv with a 'Symbol' column.

        Args:
            use_cache: Whether to use cached ticker list (24 hours)

        Returns:
            List of ticker symbols with .NS suffix
        """
        cache_file = config.DATA_DIR / "nifty500_tickers_cache.csv"
        cache_valid = False
        
        # Check cache validity
        if use_cache and cache_file.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=24):
                    cache_valid = True
            except Exception:
                pass
        
        # Load from cache if valid
        if cache_valid:
            try:
                df = pd.read_csv(cache_file)
                if "Symbol" in df.columns or "symbol" in df.columns:
                    symbol_col = "Symbol" if "Symbol" in df.columns else "symbol"
                    tickers = df[symbol_col].astype(str).tolist()
                    # Append .NS suffix if not already present
                    tickers = [f"{t}.NS" if not t.endswith(".NS") else t for t in tickers]
                    logger.info(f"Loaded {len(tickers)} tickers from cache")
                    return tickers
            except Exception as e:
                logger.warning(f"Error loading cached tickers: {e}")
        
        # Fetch from NSE website with retry logic
        # Try multiple possible URLs as NSE may change endpoints or block requests
        nse_urls = [
            "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
            "https://archives.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
            "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        ]
        
        # Retry logic for network issues
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/csv,application/csv,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        # Try each URL
        for url_idx, nse_url in enumerate(nse_urls):
            try:
                logger.info(f"Trying NSE URL {url_idx + 1}/{len(nse_urls)}: {nse_url}")
                # Reduced timeout to fail faster and try next URL
                response = requests.get(nse_url, timeout=15, headers=headers)
                response.raise_for_status()
                
                # Parse CSV from response
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                # NSE CSV typically has 'Symbol' column
                if "Symbol" in df.columns:
                    symbol_col = "Symbol"
                elif "symbol" in df.columns:
                    symbol_col = "symbol"
                elif len(df.columns) > 0:
                    # Fallback: use first column
                    symbol_col = df.columns[0]
                else:
                    raise ValueError("No symbol column found in NSE CSV")
                
                tickers = df[symbol_col].astype(str).str.strip().tolist()
                
                # Append .NS suffix for yfinance
                tickers = [f"{t}.NS" if not t.endswith(".NS") else t for t in tickers]
                
                # Save to cache
                try:
                    df.to_csv(cache_file, index=False)
                    logger.info(f"Cached {len(tickers)} tickers to {cache_file}")
                except Exception as e:
                    logger.warning(f"Could not save ticker cache: {e}")
                
                logger.info(f"Successfully fetched {len(tickers)} tickers from NSE")
                return tickers
                
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"URL {url_idx + 1} failed: {e}")
                if url_idx < len(nse_urls) - 1:
                    logger.info(f"Trying next URL...")
                    continue
                else:
                    logger.warning("All NSE URLs failed. Using fallback ticker list.")
                    return config.NIFTY_500_TICKERS
            except requests.RequestException as e:
                logger.warning(f"URL {url_idx + 1} error: {e}")
                if url_idx < len(nse_urls) - 1:
                    logger.info(f"Trying next URL...")
                    continue
                else:
                    logger.warning(f"All NSE URLs failed: {e}. Using fallback ticker list.")
                    return config.NIFTY_500_TICKERS
            except Exception as e:
                logger.error(f"Error parsing Nifty 500 list from URL {url_idx + 1}: {e}")
                if url_idx < len(nse_urls) - 1:
                    logger.info(f"Trying next URL...")
                    continue
                else:
                    logger.info("Falling back to default ticker list")
                    return config.NIFTY_500_TICKERS
        
        # Should not reach here, but fallback just in case
        logger.warning("All URL attempts exhausted. Using fallback ticker list.")
        return config.NIFTY_500_TICKERS

    def _save_market_intelligence(self, symbol: str, data: Dict) -> None:
        """
        Atomically save market intelligence data to database.
        
        Args:
            symbol: Market symbol (e.g., "^NSEI", "^INDIAVIX")
            data: Dictionary with market intelligence data
        """
        try:
            with self._db_connection() as conn:
                try:
                    if not self.use_postgres:
                        if not self.use_postgres:
                        conn.execute("BEGIN TRANSACTION")
                    cursor = conn.cursor()
                    placeholder = self._get_placeholder()
                    if self.use_postgres:
                        cursor.execute(f"""
                            INSERT INTO market_intelligence
                            (symbol, data_json, last_updated)
                            VALUES ({placeholder}, {placeholder}, {placeholder})
                            ON CONFLICT (symbol) DO UPDATE SET
                            data_json = EXCLUDED.data_json,
                            last_updated = EXCLUDED.last_updated
                        """, (
                            symbol,
                            json.dumps(data),
                            datetime.now().isoformat()
                        ))
                    else:
                        cursor.execute(f"""
                            INSERT OR REPLACE INTO market_intelligence
                            (symbol, data_json, last_updated)
                            VALUES ({placeholder}, {placeholder}, {placeholder})
                        """, (
                            symbol,
                            json.dumps(data),
                            datetime.now().isoformat()
                        ))
                    conn.commit()
                    logger.debug(f"Saved market intelligence for {symbol}")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to save market intelligence for {symbol}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error saving market intelligence for {symbol}: {e}")

    def _load_market_intelligence(self, symbol: str, max_age_hours: int = 1) -> Optional[Dict]:
        """
        Load market intelligence data from database cache.
        
        Args:
            symbol: Market symbol (e.g., "^NSEI", "^INDIAVIX")
            max_age_hours: Maximum age in hours before considering stale (default: 1 hour)
            
        Returns:
            Dictionary with market intelligence data if cache is valid, None otherwise
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                placeholder = self._get_placeholder()
                cursor.execute(f"""
                    SELECT data_json, last_updated
                    FROM market_intelligence
                    WHERE symbol = {placeholder}
                """, (symbol,))
                result = cursor.fetchone()
                
                if result:
                    data_json, last_updated_str = result
                    last_updated = datetime.fromisoformat(last_updated_str)
                    age = datetime.now() - last_updated
                    
                    if age < timedelta(hours=max_age_hours):
                        logger.debug(f"Loaded market intelligence for {symbol} from cache")
                        return json.loads(data_json)
                    else:
                        logger.debug(f"Market intelligence for {symbol} is stale ({age})")
                        return None
        except Exception as e:
            logger.debug(f"Error loading market intelligence for {symbol}: {e}")
        return None

    def get_last_update_date(self) -> Optional[datetime]:
        """
        Get the most recent last_updated timestamp from the database.
        
        Returns:
            datetime object of the most recent update, or None if no data exists
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Get the most recent update from OHLCV data
                cursor.execute("""
                    SELECT MAX(last_updated) 
                    FROM ohlcv_data
                """)
                result = cursor.fetchone()
                
                if result and result[0]:
                    return datetime.fromisoformat(result[0])
        except Exception as e:
            logger.debug(f"Error getting last update date: {e}")
        
        return None

    def _calculate_data_hash(self, ticker: str, df: pd.DataFrame) -> str:
        """
        Calculate a hash of the data to detect changes.
        
        Args:
            ticker: Stock ticker
            df: DataFrame with OHLCV data
            
        Returns:
            Hash string representing the data state
        """
        import hashlib
        # Use last date and last close price as hash components
        if df.empty:
            return hashlib.md5(f"{ticker}_empty".encode()).hexdigest()
        
        last_row = df.iloc[-1]
        hash_str = f"{ticker}_{df.index[-1]}_{last_row['close']:.2f}_{len(df)}"
        return hashlib.md5(hash_str.encode()).hexdigest()

    def save_strategy_results(
        self, 
        ticker: str, 
        regime: str, 
        signals: List[Dict], 
        data_hash: str
    ) -> None:
        """
        Save strategy results (trade signals) to database cache.
        
        Args:
            ticker: Stock ticker symbol
            regime: Market regime ("AGGRESSIVE", "DEFENSIVE", "CASH_PROTECTION")
            signals: List of signal dictionaries (from TradeSignal.model_dump())
            data_hash: Hash of the input data to detect changes
        """
        try:
            with self._db_connection() as conn:
                try:
                    if not self.use_postgres:
                        if not self.use_postgres:
                        conn.execute("BEGIN TRANSACTION")
                    cursor = conn.cursor()
                    placeholder = self._get_placeholder()
                    if self.use_postgres:
                        cursor.execute(f"""
                            INSERT INTO strategy_results
                            (ticker, regime, signals_json, data_hash, last_updated)
                            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                            ON CONFLICT (ticker, regime, data_hash) DO UPDATE SET
                            signals_json = EXCLUDED.signals_json,
                            last_updated = EXCLUDED.last_updated
                        """, (
                            ticker,
                            regime,
                            json.dumps(signals),
                            data_hash,
                            datetime.now().isoformat()
                        ))
                    else:
                        cursor.execute(f"""
                            INSERT OR REPLACE INTO strategy_results
                            (ticker, regime, signals_json, data_hash, last_updated)
                            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                        """, (
                            ticker,
                            regime,
                            json.dumps(signals),
                            data_hash,
                            datetime.now().isoformat()
                        ))
                    conn.commit()
                    logger.debug(f"Saved strategy results for {ticker} ({regime})")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to save strategy results for {ticker}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error saving strategy results for {ticker}: {e}")

    def load_strategy_results(
        self, 
        ticker: str, 
        regime: str, 
        data_hash: str,
        max_age_hours: int = 24
    ) -> Optional[List[Dict]]:
        """
        Load strategy results from database cache if they match current data.
        
        Args:
            ticker: Stock ticker symbol
            regime: Market regime
            data_hash: Hash of current data to verify cache validity
            max_age_hours: Maximum age in hours before considering stale (default: 24 hours)
            
        Returns:
            List of signal dictionaries if cache is valid, None otherwise
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                placeholder = self._get_placeholder()
                cursor.execute(f"""
                    SELECT signals_json, data_hash, last_updated
                    FROM strategy_results
                    WHERE ticker = {placeholder} AND regime = {placeholder}
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, (ticker, regime))
                result = cursor.fetchone()
                
                if result:
                    signals_json, cached_hash, last_updated_str = result
                    last_updated = datetime.fromisoformat(last_updated_str)
                    age = datetime.now() - last_updated
                    
                    # Check if data hash matches (data hasn't changed)
                    if cached_hash == data_hash and age < timedelta(hours=max_age_hours):
                        logger.debug(f"Loaded strategy results for {ticker} ({regime}) from cache")
                        return json.loads(signals_json)
                    else:
                        if cached_hash != data_hash:
                            logger.debug(f"Strategy cache for {ticker} invalidated (data changed)")
                            # Auto-delete stale cache entry
                            placeholder = self._get_placeholder()
                            cursor.execute(f"""
                                DELETE FROM strategy_results
                                WHERE ticker = {placeholder} AND regime = {placeholder} AND data_hash = {placeholder}
                            """, (ticker, regime, cached_hash))
                            conn.commit()
                        else:
                            logger.debug(f"Strategy cache for {ticker} is stale ({age})")
                        return None
        except Exception as e:
            logger.debug(f"Error loading strategy results for {ticker}: {e}")
        return None

    def clear_strategy_cache(self, ticker: Optional[str] = None, regime: Optional[str] = None) -> None:
        """
        Clear strategy results cache.
        
        Args:
            ticker: Optional ticker to clear (if None, clears all)
            regime: Optional regime to clear (if None, clears all)
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                placeholder = self._get_placeholder()
                if ticker and regime:
                    cursor.execute(f"""
                        DELETE FROM strategy_results
                        WHERE ticker = {placeholder} AND regime = {placeholder}
                    """, (ticker, regime))
                elif ticker:
                    cursor.execute(f"""
                        DELETE FROM strategy_results
                        WHERE ticker = {placeholder}
                    """, (ticker,))
                elif regime:
                    cursor.execute(f"""
                        DELETE FROM strategy_results
                        WHERE regime = {placeholder}
                    """, (regime,))
                else:
                    cursor.execute("DELETE FROM strategy_results")
                conn.commit()
                logger.info(f"Cleared strategy cache for ticker={ticker}, regime={regime}")
        except Exception as e:
            logger.error(f"Error clearing strategy cache: {e}")

    def load_nifty_500_tickers(self, csv_path: str | None = None) -> List[str]:
        """
        Legacy method: Load Nifty 500 tickers from CSV or return default list.
        For new code, use get_nifty500_tickers() instead.

        Args:
            csv_path: Optional path to CSV file

        Returns:
            List of ticker symbols
        """
        # Use dynamic fetcher by default
        return self.get_nifty500_tickers(use_cache=True)

