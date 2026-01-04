"""Company Enrichment Engine: Fetch and cache fundamental metadata."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompanyEnrichment:
    """
    Lazy-loading company metadata fetcher with intelligent caching.
    Only fetches data for top filtered results to optimize performance.
    """

    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize enrichment engine.

        Args:
            cache_file: Path to cache file (default: data/company_cache.json)
        """
        if cache_file is None:
            cache_file = config.DATA_DIR / "company_cache.json"
        
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = {}
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
        
        # Load existing cache
        self._load_cache()
        
        # Static sector map for fast lookup (major Nifty 500 stocks)
        self._sector_map = self._build_static_sector_map()

    def _build_static_sector_map(self) -> Dict[str, str]:
        """
        Build static sector map for fast lookup without API calls.
        
        Returns:
            Dictionary mapping ticker -> sector
        """
        # Major Nifty 500 stocks with their sectors
        sector_map = {
            # IT Sector
            "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT",
            "TECHM.NS": "IT", "LTIM.NS": "IT", "MPHASIS.NS": "IT", "COFORGE.NS": "IT",
            
            # Banking
            "HDFCBANK.NS": "Bank", "ICICIBANK.NS": "Bank", "SBIN.NS": "Bank",
            "KOTAKBANK.NS": "Bank", "AXISBANK.NS": "Bank", "INDUSINDBK.NS": "Bank",
            "PNB.NS": "Bank", "BANKBARODA.NS": "Bank", "UNIONBANK.NS": "Bank",
            
            # Auto
            "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto",
            "BAJAJ-AUTO.NS": "Auto", "HEROMOTOCO.NS": "Auto", "EICHERMOT.NS": "Auto",
            "ASHOKLEY.NS": "Auto", "TVSMOTOR.NS": "Auto",
            
            # Pharma
            "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
            "LUPIN.NS": "Pharma", "TORNTPHARM.NS": "Pharma", "DIVISLAB.NS": "Pharma",
            "GLENMARK.NS": "Pharma", "AUROPHARMA.NS": "Pharma",
            
            # Energy
            "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "IOC.NS": "Energy",
            "BPCL.NS": "Energy", "HINDPETRO.NS": "Energy", "GAIL.NS": "Energy",
            
            # Metal
            "TATASTEEL.NS": "Metal", "JSWSTEEL.NS": "Metal", "HINDALCO.NS": "Metal",
            "VEDL.NS": "Metal", "SAIL.NS": "Metal", "JINDALSAW.NS": "Metal",
            
            # FMCG
            "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
            "BRITANNIA.NS": "FMCG", "DABUR.NS": "FMCG", "GODREJCP.NS": "FMCG",
            "MARICO.NS": "FMCG", "TATACONSUM.NS": "FMCG",
            
            # Telecom
            "BHARTIARTL.NS": "Telecom",
            
            # Cement
            "ULTRACEMCO.NS": "Cement", "SHREECEM.NS": "Cement", "ACC.NS": "Cement",
            "AMBUJACEM.NS": "Cement", "RAMCOCEM.NS": "Cement",
            
            # Retail
            "TRENT.NS": "Retail", "SHOPPERSSTOP.NS": "Retail", "V-MART.NS": "Retail",
            
            # Finance
            "BAJFINANCE.NS": "Finance", "BAJAJFINSV.NS": "Finance", "HDFC.NS": "Finance",
            "LICI.NS": "Finance", "SBILIFE.NS": "Finance",
        }
        
        logger.info(f"Built static sector map with {len(sector_map)} tickers")
        return sector_map

    def get_sector_fast(self, ticker: str) -> str:
        """
        Fast sector lookup using static map (no API call).

        Args:
            ticker: Stock ticker

        Returns:
            Sector name or "Unknown"
        """
        return self._sector_map.get(ticker, "Unknown")

    def get_sector_performance(
        self, sector: str, data_engine, lookback_days: int = 5
    ) -> Dict[str, float]:
        """
        Calculate sector performance (% change) for swing trading.
        
        Args:
            sector: Sector name
            data_engine: FastDataEngine instance for fetching sector data
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with 'change_pct' and 'status' (bullish/bearish/neutral)
        """
        # Map sector names to sector indices (simplified - can be expanded)
        sector_indices = {
            "Auto": "^CNXAUTO",
            "IT": "^CNXIT",
            "Bank": "^CNXBAN",
            "Pharma": "^CNXPHR",
            "Metal": "^CNXMETAL",
            "Energy": "^CNXENERGY",
            "FMCG": "^CNXFMCG",
            "Realty": "^CNXREALTY",
        }
        
        sector_symbol = sector_indices.get(sector)
        if not sector_symbol or not data_engine:
            return {"change_pct": 0.0, "status": "neutral"}
        
        try:
            # Fetch sector index data
            sector_df = data_engine.fetch_single(sector_symbol, period="1mo", use_cache=True)
            if sector_df.empty or len(sector_df) < lookback_days:
                return {"change_pct": 0.0, "status": "neutral"}
            
            # Calculate % change over lookback period
            current_price = sector_df["close"].iloc[-1]
            past_price = sector_df["close"].iloc[-lookback_days] if len(sector_df) >= lookback_days else sector_df["close"].iloc[0]
            
            change_pct = ((current_price - past_price) / past_price) * 100
            
            # Determine status
            if change_pct > 1.0:
                status = "bullish"
            elif change_pct < -1.0:
                status = "bearish"
            else:
                status = "neutral"
            
            return {
                "change_pct": round(change_pct, 2),
                "status": status,
            }
        except Exception as e:
            logger.warning(f"Error calculating sector performance for {sector}: {e}")
            return {"change_pct": 0.0, "status": "neutral"}

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} company profiles from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached data is still valid."""
        if "cached_at" not in cached_data:
            return False
        
        cached_time = datetime.fromisoformat(cached_data["cached_at"])
        expiry_time = cached_time + timedelta(hours=self.cache_expiry_hours)
        
        return datetime.now() < expiry_time

    def get_company_profile(self, ticker: str) -> Dict[str, str]:
        """
        Get company profile with lazy loading and caching.

        Args:
            ticker: Stock ticker (e.g., "RELIANCE.NS")

        Returns:
            Dictionary with company metadata:
            - name: Company long name
            - sector: Business sector
            - market_cap: Formatted market cap (₹Cr)
            - pe_ratio: Trailing P/E ratio
            - summary: Business summary (truncated to 150 chars)
        """
        # Check cache first
        if ticker in self.cache:
            cached_data = self.cache[ticker]
            if self._is_cache_valid(cached_data):
                logger.debug(f"Using cached profile for {ticker}")
                return cached_data

        # Fetch from yfinance
        try:
            logger.info(f"Fetching company profile for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract and format data
            long_name = info.get("longName", info.get("shortName", ticker))
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            
            # Format market cap
            market_cap = info.get("marketCap", 0)
            if market_cap:
                market_cap_cr = market_cap / 10_000_000_000  # Convert to Cr (10^10)
                if market_cap_cr >= 1000:
                    market_cap_str = f"₹{market_cap_cr/100:.1f}L Cr"
                else:
                    market_cap_str = f"₹{market_cap_cr:.1f} Cr"
            else:
                market_cap_str = "N/A"

            # P/E ratio
            pe_ratio = info.get("trailingPE", None)
            pe_str = f"{pe_ratio:.1f}" if pe_ratio else "N/A"

            # Business summary (truncate to 150 chars)
            summary = info.get("longBusinessSummary", "")
            if summary:
                summary = summary[:150] + "..." if len(summary) > 150 else summary
            else:
                summary = "No description available."

            # Market cap category
            if market_cap:
                if market_cap >= 200_000_000_000:  # >= 20,000 Cr
                    cap_category = "Large Cap"
                elif market_cap >= 50_000_000_000:  # >= 5,000 Cr
                    cap_category = "Mid Cap"
                else:
                    cap_category = "Small Cap"
            else:
                cap_category = "Unknown"

            profile = {
                "name": long_name,
                "sector": sector,
                "industry": industry,
                "market_cap": market_cap_str,
                "market_cap_value": market_cap,  # Raw value for sorting
                "cap_category": cap_category,
                "pe_ratio": pe_str,
                "pe_value": pe_ratio if pe_ratio else 0,
                "summary": summary,
                "cached_at": datetime.now().isoformat(),
            }

            # Update cache
            self.cache[ticker] = profile
            self._save_cache()

            return profile

        except Exception as e:
            logger.error(f"Error fetching profile for {ticker}: {e}")
            # Return default profile
            return {
                "name": ticker.replace(".NS", ""),
                "sector": "Unknown",
                "industry": "Unknown",
                "market_cap": "N/A",
                "market_cap_value": 0,
                "cap_category": "Unknown",
                "pe_ratio": "N/A",
                "pe_value": 0,
                "summary": "Unable to fetch company information.",
                "cached_at": datetime.now().isoformat(),
            }

    def get_batch_profiles(self, tickers: list[str]) -> Dict[str, Dict[str, str]]:
        """
        Get profiles for multiple tickers (lazy loading with cache).

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker -> profile
        """
        profiles = {}
        for ticker in tickers:
            profiles[ticker] = self.get_company_profile(ticker)
        
        return profiles

    def get_top_sector(self, tickers: list[str]) -> str:
        """
        Determine the top sector from a list of tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Top sector name (e.g., "Technology", "Auto")
        """
        sector_counts: Dict[str, int] = {}
        
        for ticker in tickers:
            profile = self.get_company_profile(ticker)
            sector = profile.get("sector", "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        if not sector_counts:
            return "Unknown"
        
        top_sector = max(sector_counts.items(), key=lambda x: x[1])[0]
        return top_sector

