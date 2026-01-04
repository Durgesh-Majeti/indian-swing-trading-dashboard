"""The Weekly Climber Strategy - Multi-Timeframe."""

from typing import Optional, Dict
import pandas as pd
import pandas_ta as ta
from src.engine.strategy_base import BaseStrategy
from src.engine.risk import SwingRiskEngine
from src.models.schemas import TradeTicket
import config


class WeeklyClimberStrategy(BaseStrategy):
    """
    Strategy: The Weekly Climber (Multi-Timeframe).
    
    Concept: Daily entry aligned with Weekly power.
    Rules:
    - Weekly Chart: MACD Line > Signal Line
    - Daily Chart: Price crosses above 20 SMA
    - Filter: Daily RSI is 50-65 (Rising room)
    """

    def get_name(self) -> str:
        """Get strategy name."""
        return "The Weekly Climber"

    def get_description(self) -> str:
        """Get strategy description."""
        return "Multi-Timeframe: Weekly MACD bullish, Daily crosses 20 SMA, RSI 50-65"

    def get_default_params(self) -> Dict:
        """Get default parameters."""
        return {
            "rsi_min": config.WEEKLY_CLIMBER_RSI_MIN,
            "rsi_max": config.WEEKLY_CLIMBER_RSI_MAX,
        }

    def is_allowed_in_regime(self, regime: str) -> bool:
        """Check if strategy is allowed in regime."""
        return regime in ["AGGRESSIVE", "DEFENSIVE"]

    def get_strategy_group(self) -> str:
        """Get strategy group."""
        return "C"  # Momentum

    def scan(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime: str = "AGGRESSIVE",
        params: Optional[Dict] = None,
    ) -> Optional[TradeTicket]:
        """Scan for Weekly Climber setup."""
        # Merge with defaults to ensure all required params are present
        default_params = self.get_default_params()
        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        if df.empty or len(df) < 50:
            return None

        # Resample daily data to weekly (5-day aggregation)
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Resample to weekly (W = week ending Sunday)
            weekly_df = df.resample("W").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

            if weekly_df.empty or len(weekly_df) < 10:
                return None

            # Weekly Chart: MACD Line > Signal Line
            macd_weekly = ta.macd(weekly_df["close"])
            if macd_weekly is None or len(macd_weekly) < 2:
                return None

            if isinstance(macd_weekly, pd.DataFrame):
                macd_line = macd_weekly.iloc[:, 0]  # MACD line
                signal_line = macd_weekly.iloc[:, 1]  # Signal line
            else:
                return None

            weekly_macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

            # Daily Chart: Price crosses above 20 SMA
            df["sma_20"] = ta.sma(df["close"], length=20)
            current = df.iloc[-1]
            prev = df.iloc[-2]

            price_crosses_sma20 = (
                current["close"] > current["sma_20"] and
                prev["close"] <= prev["sma_20"]
            )

            # Filter: Daily RSI is 50-65 (Rising room)
            rsi = ta.rsi(df["close"], length=14)
            if rsi.empty or len(rsi) == 0:
                return None

            current_rsi = rsi.iloc[-1]
            rsi_in_range = (
                not pd.isna(current_rsi) and
                params["rsi_min"] <= current_rsi <= params["rsi_max"]
            )

            conditions = [
                weekly_macd_bullish,
                price_crosses_sma20,
                rsi_in_range,
            ]

            if all(conditions):
                entry_price = current["close"]
                
                # Get swing low for stop loss
                risk_engine = SwingRiskEngine()
                swing_levels = risk_engine.get_swing_levels(df, lookback=20)
                swing_low = swing_levels["swing_low"]
                
                stop_loss = risk_engine.calculate_swing_stop_loss(
                    strategy_group="C",
                    df=df,
                    entry_price=entry_price,
                    swing_low=swing_low,
                )
                
                targets = risk_engine.calculate_swing_targets(df, entry_price, stop_loss)
                target = targets["target"]

                confidence = 0.80  # Good confidence for multi-timeframe alignment

                metadata = {
                    "sma_20": float(current["sma_20"]),
                    "rsi": float(current_rsi),
                    "weekly_macd": float(macd_line.iloc[-1]),
                    "weekly_signal": float(signal_line.iloc[-1]),
                    "holding_period_days": 15,
                    "target_type": "momentum",
                    "strategy_group": "C",
                    "target_1": targets["target_1"],
                    "target_2": targets["target_2"],
                }

                return self.trade_assistant.create_trade_ticket(
                    ticker=ticker,
                    strategy_name="The Weekly Climber",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=confidence,
                    metadata=metadata,
                )

        except Exception:
            pass

        return None

