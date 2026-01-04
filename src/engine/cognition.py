"""Cognition Engine: Market Intelligence and Regime Diagnosis."""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from src.feed.data import FastDataEngine
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitionEngine:
    """
    Market Intelligence Engine that diagnoses:
    - Market Regime (Bull/Correction/Bear)
    - Volatility Regime (VIX-based)
    - Sector Intelligence (Heatmap)
    """

    def __init__(self, data_engine: FastDataEngine):
        """
        Initialize cognition engine.

        Args:
            data_engine: Fast data engine for fetching market data
        """
        self.data_engine = data_engine
        self._regime_cache: Optional[Dict] = None
        self._sector_cache: Optional[Dict] = None

    def diagnose_market_regime(self, index_symbol: str = "^NSEI", force_refresh: bool = False) -> Dict[str, str]:
        """
        Diagnose market regime based on Nifty 50 position relative to EMAs.

        Args:
            index_symbol: Index symbol (default: ^NSEI for Nifty 50)
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            Dictionary with:
            - regime: "AGGRESSIVE" | "DEFENSIVE" | "CASH_PROTECTION"
            - status: Human-readable status
            - price: Current price
            - ema_50: 50 EMA value
            - ema_200: 200 EMA value
        """
        try:
            # Try to load from market intelligence cache first
            if not force_refresh:
                cached = self.data_engine._load_market_intelligence(index_symbol, max_age_hours=1)
                if cached:
                    logger.debug(f"Using cached market regime for {index_symbol}")
                    return cached
            
            df = self.data_engine.fetch_single(index_symbol, period="1y", use_cache=True, force_refresh=force_refresh)
            
            if df.empty or len(df) < 200:
                logger.warning(f"Insufficient data for regime diagnosis: {index_symbol}")
                return {
                    "regime": "UNKNOWN",
                    "status": "Insufficient data",
                    "price": 0.0,
                    "ema_50": 0.0,
                    "ema_200": 0.0,
                }

            # Calculate EMAs
            df["ema_50"] = ta.ema(df["close"], length=50)
            df["ema_200"] = ta.ema(df["close"], length=200)

            # Get current values (handle weekend)
            current_price = df["close"].iloc[-1]
            ema_50 = df["ema_50"].iloc[-1]
            ema_200 = df["ema_200"].iloc[-1]

            # Calculate % change (today vs yesterday)
            if len(df) >= 2:
                previous_close = df["close"].iloc[-2]
                nifty_50_change = ((current_price - previous_close) / previous_close) * 100
            else:
                nifty_50_change = 0.0
            
            # Diagnose regime
            if current_price > ema_50:
                regime = "AGGRESSIVE"
                status = "🟢 BULL - Price above 50 EMA"
            elif current_price > ema_200:
                regime = "DEFENSIVE"
                status = "🟡 CORRECTION - Price between 50 & 200 EMA"
            else:
                regime = "CASH_PROTECTION"
                status = "🔴 BEAR - Price below 200 EMA"

            result = {
                "regime": regime,
                "status": status,
                "price": round(current_price, 2),
                "ema_50": round(ema_50, 2),
                "ema_200": round(ema_200, 2),
                "nifty_50_change": round(nifty_50_change, 2),
            }

            logger.info(f"Market Regime: {regime} - {status}")
            
            # Save to market intelligence cache
            if not force_refresh:
                self.data_engine._save_market_intelligence(index_symbol, result)
            
            return result

        except Exception as e:
            logger.error(f"Error diagnosing market regime: {e}")
            return {
                "regime": "UNKNOWN",
                "status": f"Error: {str(e)}",
                "price": 0.0,
                "ema_50": 0.0,
                "ema_200": 0.0,
            }

    def diagnose_volatility_regime(self, vix_symbol: str = "^INDIAVIX", force_refresh: bool = False) -> Dict[str, any]:
        """
        Diagnose volatility regime based on India VIX.

        Args:
            vix_symbol: VIX symbol (default: ^INDIAVIX)
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            Dictionary with:
            - vix_value: Current VIX value
            - regime: "PANIC" | "NORMAL" | "CALM"
            - position_multiplier: Position size multiplier (0.5 for panic, 1.0 for normal)
            - status: Human-readable status
        """
        try:
            # Try to load from market intelligence cache first
            if not force_refresh:
                cached = self.data_engine._load_market_intelligence(vix_symbol, max_age_hours=1)
                if cached:
                    logger.debug(f"Using cached volatility regime for {vix_symbol}")
                    return cached
            
            df = self.data_engine.fetch_single(vix_symbol, period="3mo", use_cache=True, force_refresh=force_refresh)
            
            if df.empty:
                logger.warning(f"No VIX data available: {vix_symbol}")
                return {
                    "vix_value": 0.0,
                    "regime": "UNKNOWN",
                    "position_multiplier": 1.0,
                    "status": "VIX data unavailable",
                }

            current_vix = df["close"].iloc[-1]

            if current_vix > 22:
                regime = "PANIC"
                position_multiplier = 0.5  # Halve position sizes
                status = f"⚠️ PANIC - VIX {current_vix:.1f} (Halve positions)"
            elif current_vix < 18:
                regime = "CALM"
                position_multiplier = 1.0
                status = f"✅ CALM - VIX {current_vix:.1f} (Normal sizing)"
            else:
                regime = "NORMAL"
                position_multiplier = 1.0
                status = f"📊 NORMAL - VIX {current_vix:.1f} (Normal sizing)"

            result = {
                "vix": round(current_vix, 2),  # Changed from vix_value to vix for consistency
                "vix_value": round(current_vix, 2),  # Keep both for backward compatibility
                "regime": regime,
                "position_multiplier": position_multiplier,
                "status": status,
            }

            logger.info(f"Volatility Regime: {regime} - {status}")
            
            # Save to market intelligence cache
            if not force_refresh:
                self.data_engine._save_market_intelligence(vix_symbol, result)
            
            return result

        except Exception as e:
            logger.error(f"Error diagnosing volatility regime: {e}")
            return {
                "vix_value": 0.0,
                "regime": "UNKNOWN",
                "position_multiplier": 1.0,
                "status": f"Error: {str(e)}",
            }

    def build_sector_heatmap(
        self,
        sector_indices: Optional[List[Tuple[str, str]]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build sector heatmap by calculating daily % change for major sectors.

        Args:
            sector_indices: List of (symbol, name) tuples. Defaults to major Indian sectors.
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            Dictionary mapping sector name to:
            - change_pct: Daily percentage change
            - status: "LEADING" | "LAGGING" | "NEUTRAL"
        """
        if sector_indices is None:
            sector_indices = [
                ("^CNXAUTO", "Auto"),
                ("^CNXIT", "IT"),
                ("^CNXBAN", "Bank"),
                ("^CNXPHR", "Pharma"),
                ("^CNXMETAL", "Metal"),
                ("^CNXFMCG", "FMCG"),
                ("^CNXENERGY", "Energy"),
            ]

        heatmap = {}
        changes = []

        for symbol, name in sector_indices:
            try:
                df = self.data_engine.fetch_single(symbol, period="1mo", use_cache=True, force_refresh=force_refresh)
                
                if df.empty or len(df) < 2:
                    logger.warning(f"Insufficient data for sector: {name}")
                    continue

                # Calculate daily % change
                current = df["close"].iloc[-1]
                previous = df["close"].iloc[-2]
                change_pct = ((current - previous) / previous) * 100

                heatmap[name] = {
                    "change_pct": round(change_pct, 2),
                    "price": round(current, 2),
                }
                changes.append((name, change_pct))

            except Exception as e:
                logger.error(f"Error fetching sector {name}: {e}")
                continue

        # Identify leading and lagging sectors
        if changes:
            changes.sort(key=lambda x: x[1], reverse=True)
            leading = changes[0][0] if changes[0][1] > 0.5 else None
            lagging = changes[-1][0] if changes[-1][1] < -0.5 else None

            # Tag sectors
            for name in heatmap:
                if name == leading:
                    heatmap[name]["status"] = "LEADING"
                elif name == lagging:
                    heatmap[name]["status"] = "LAGGING"
                else:
                    heatmap[name]["status"] = "NEUTRAL"

        logger.info(f"Built sector heatmap with {len(heatmap)} sectors")
        return heatmap

    def get_all_intelligence(self, force_refresh: bool = False) -> Dict:
        """
        Get complete market intelligence: Regime + Volatility + Sectors.

        Args:
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            Complete intelligence dictionary
        """
        market_regime = self.diagnose_market_regime(force_refresh=force_refresh)
        volatility_regime = self.diagnose_volatility_regime(force_refresh=force_refresh)
        sector_heatmap = self.build_sector_heatmap(force_refresh=force_refresh)

        # Find top sector
        top_sector = None
        top_change = -999
        for name, data in sector_heatmap.items():
            if data["change_pct"] > top_change:
                top_change = data["change_pct"]
                top_sector = name

        intelligence = {
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "sector_heatmap": sector_heatmap,
            "sector_intelligence": {  # Add sector_intelligence for dashboard compatibility
                "top_sector": top_sector or "N/A",
                "top_sector_change": round(top_change, 2) if top_sector else 0.0,
            },
            "top_sector": {  # Keep for backward compatibility
                "name": top_sector,
                "change_pct": top_change,
            } if top_sector else None,
        }

        return intelligence

