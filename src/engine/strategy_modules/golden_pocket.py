"""The Golden Pocket Strategy - Fibonacci Retracement."""

from typing import Optional, Dict
import pandas as pd
import pandas_ta as ta
from src.engine.strategy_base import BaseStrategy
from src.engine.risk import SwingRiskEngine
from src.models.schemas import TradeTicket
import config


class GoldenPocketStrategy(BaseStrategy):
    """
    Strategy: The Golden Pocket (Fibonacci).
    
    Concept: Algo buying at deep discounts.
    Rules:
    - Trend: Price > 200 EMA
    - Math: Calculate Fib Retracement from last Swing Low to Swing High
    - Trigger: Price touches 0.618 (61.8%) Level and bounces
    """

    def get_name(self) -> str:
        """Get strategy name."""
        return "The Golden Pocket"

    def get_description(self) -> str:
        """Get strategy description."""
        return "Fibonacci Retracement: Price > 200 EMA, touches 0.618 Fib level and bounces"

    def get_default_params(self) -> Dict:
        """Get default parameters."""
        return {
            "fib_level": config.GOLDEN_POCKET_FIB_LEVEL,
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
        """Scan for Golden Pocket setup."""
        # Merge with defaults to ensure all required params are present
        default_params = self.get_default_params()
        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        if df.empty or len(df) < 200:
            return None

        # Calculate 200 EMA for trend filter
        df["ema_200"] = ta.ema(df["close"], length=200)
        current = df.iloc[-1]

        # Trend: Price > 200 EMA
        price_above_200 = current["close"] > current["ema_200"]
        if not price_above_200 or pd.isna(current["ema_200"]):
            return None

        # Get swing levels
        risk_engine = SwingRiskEngine()
        swing_levels = risk_engine.get_swing_levels(df, lookback=50)
        swing_high = swing_levels["swing_high"]
        swing_low = swing_levels["swing_low"]

        # Calculate Fibonacci levels
        fib_levels = risk_engine.calculate_fibonacci_levels(swing_low, swing_high)
        fib_618 = fib_levels["fib_0_618"]

        # Trigger: Price touches 0.618 level (within 1%) and bounces
        distance_from_fib = abs(current["close"] - fib_618) / fib_618
        touches_fib = distance_from_fib <= 0.01

        # Bounce: Current close is above the low (bounced from fib level)
        current_low = current["low"]
        bounced = current["close"] > current_low and current_low <= fib_618 * 1.01

        conditions = [
            price_above_200,
            touches_fib or (current_low <= fib_618 * 1.01),
            bounced,
        ]

        if all(conditions):
            entry_price = current["close"]
            
            # Use SwingRiskEngine for stop loss and targets
            stop_loss = risk_engine.calculate_swing_stop_loss(
                strategy_group="A",
                df=df,
                entry_price=entry_price,
                support_level=fib_618,
            )
            
            targets = risk_engine.calculate_swing_targets(df, entry_price, stop_loss)
            target = targets["target"]

            confidence = 0.80  # Higher confidence for fib bounce

            metadata = {
                "fib_618": float(fib_618),
                "swing_high": float(swing_high),
                "swing_low": float(swing_low),
                "holding_period_days": 15,
                "target_type": "swing_high",
                "strategy_group": "A",
                "target_1": targets["target_1"],
                "target_2": targets["target_2"],
                "fib_levels": {k: float(v) for k, v in fib_levels.items()},
            }

            return self.trade_assistant.create_trade_ticket(
                ticker=ticker,
                strategy_name="The Golden Pocket",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                confidence=confidence,
                metadata=metadata,
            )

        return None

