"""The Institutional Wave Strategy - 50 EMA Support."""

from typing import Optional, Dict
import pandas as pd
import pandas_ta as ta
from src.engine.strategy_base import BaseStrategy
from src.engine.risk import SwingRiskEngine
from src.models.schemas import TradeTicket
import config


class InstitutionalWaveStrategy(BaseStrategy):
    """
    Strategy: The Institutional Wave (50 EMA).
    
    Concept: Mutual Funds defend the 50-Day line.
    Rules:
    - Trend: Price > 200 EMA
    - Zone: Price touches or dips slightly below 50 EMA
    - Trigger: Close > Open (Green Candle) while near 50 EMA
    """

    def get_name(self) -> str:
        """Get strategy name."""
        return "The Institutional Wave"

    def get_description(self) -> str:
        """Get strategy description."""
        return "50 EMA Support: Price > 200 EMA, touches 50 EMA, Green Candle"

    def get_default_params(self) -> Dict:
        """Get default parameters."""
        return {
            "ema_50_period": config.INSTITUTIONAL_WAVE_EMA_50,
            "ema_200_period": config.INSTITUTIONAL_WAVE_EMA_200,
        }

    def is_allowed_in_regime(self, regime: str) -> bool:
        """Check if strategy is allowed in regime."""
        return regime in ["AGGRESSIVE", "DEFENSIVE"]

    def get_strategy_group(self) -> str:
        """Get strategy group."""
        return "A"  # Support

    def scan(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime: str = "AGGRESSIVE",
        params: Optional[Dict] = None,
    ) -> Optional[TradeTicket]:
        """Scan for Institutional Wave setup."""
        # Merge with defaults to ensure all required params are present
        default_params = self.get_default_params()
        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        if df.empty or len(df) < 200:
            return None

        # Calculate EMAs
        df["ema_50"] = ta.ema(df["close"], length=params["ema_50_period"])
        df["ema_200"] = ta.ema(df["close"], length=params["ema_200_period"])

        current = df.iloc[-1]

        # Trend: Price > 200 EMA
        price_above_200 = current["close"] > current["ema_200"]
        if not price_above_200 or pd.isna(current["ema_200"]):
            return None

        # Zone: Price touches or dips slightly below 50 EMA (within 1%)
        ema_50_value = current["ema_50"]
        if pd.isna(ema_50_value):
            return None

        distance_from_ema50 = abs(current["close"] - ema_50_value) / ema_50_value
        touches_ema50 = distance_from_ema50 <= 0.01  # Within 1%

        # Allow slight dip below 50 EMA (up to 0.5% below)
        dips_below = (current["low"] <= ema_50_value) and (current["low"] >= ema_50_value * 0.995)

        # Trigger: Green candle (Close > Open) while near 50 EMA
        is_green_candle = current["close"] > current["open"]

        conditions = [
            price_above_200,
            (touches_ema50 or dips_below),
            is_green_candle,
        ]

        if all(conditions):
            entry_price = current["close"]
            
            # Use SwingRiskEngine for stop loss and targets
            risk_engine = SwingRiskEngine()
            stop_loss = risk_engine.calculate_swing_stop_loss(
                strategy_group="A",
                df=df,
                entry_price=entry_price,
                support_level=ema_50_value,
            )
            
            targets = risk_engine.calculate_swing_targets(df, entry_price, stop_loss)
            target = targets["target"]

            confidence = 0.75  # Moderate confidence for support bounce

            metadata = {
                "ema_50": float(ema_50_value),
                "ema_200": float(current["ema_200"]),
                "holding_period_days": 15,  # 10-20 day range
                "target_type": "swing_high",
                "strategy_group": "A",
                "target_1": targets["target_1"],
                "target_2": targets["target_2"],
            }

            return self.trade_assistant.create_trade_ticket(
                ticker=ticker,
                strategy_name="The Institutional Wave",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                confidence=confidence,
                metadata=metadata,
            )

        return None

