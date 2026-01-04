"""The Coil Breakout Strategy - Base Building."""

from typing import Optional, Dict
import pandas as pd
import pandas_ta as ta
from src.engine.strategy_base import BaseStrategy
from src.engine.risk import SwingRiskEngine
from src.models.schemas import TradeTicket
import config


class CoilBreakoutStrategy(BaseStrategy):
    """
    Strategy: The Coil Breakout (Base Building).
    
    Concept: Expansion follows Contraction.
    Rules:
    - Base: Price range (High - Low) over last 10 days is < 5%
    - Volume: 10-day Avg Volume is declining (Drying up)
    - Trigger: Price closes above 10-Day High with Volume > 1.5x Avg
    """

    def get_name(self) -> str:
        """Get strategy name."""
        return "The Coil Breakout"

    def get_description(self) -> str:
        """Get strategy description."""
        return "Base Building: Price range < 5%, Volume declining, Breakout above 10-day high"

    def get_default_params(self) -> Dict:
        """Get default parameters."""
        return {
            "range_threshold": config.COIL_BREAKOUT_RANGE_THRESHOLD,
            "volume_multiplier": config.COIL_BREAKOUT_VOLUME_MULTIPLIER,
        }

    def is_allowed_in_regime(self, regime: str) -> bool:
        """Check if strategy is allowed in regime."""
        return regime == "AGGRESSIVE"

    def get_strategy_group(self) -> str:
        """Get strategy group."""
        return "B"  # Breakout

    def scan(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime: str = "AGGRESSIVE",
        params: Optional[Dict] = None,
    ) -> Optional[TradeTicket]:
        """Scan for Coil Breakout setup."""
        # Merge with defaults to ensure all required params are present
        default_params = self.get_default_params()
        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        if df.empty or len(df) < 20:
            return None

        current = df.iloc[-1]
        recent_10 = df.tail(10)

        # Base: Price range (High - Low) over last 10 days is < 5%
        price_range = recent_10["high"].max() - recent_10["low"].min()
        price_range_pct = price_range / current["close"]
        is_tight_base = price_range_pct < params["range_threshold"]

        # Volume: 10-day Avg Volume is declining (Drying up)
        volume_10_avg = recent_10["volume"].mean()
        volume_5_avg = df.tail(5)["volume"].mean()
        volume_declining = volume_5_avg < volume_10_avg * 0.9  # 10% decline

        # Trigger: Price closes above 10-Day High with Volume > 1.5x Avg
        high_10 = recent_10["high"].max()
        closes_above_high = current["close"] > high_10
        
        avg_volume_20 = df.tail(20)["volume"].mean()
        volume_breakout = current["volume"] >= avg_volume_20 * params["volume_multiplier"]

        conditions = [
            is_tight_base,
            volume_declining,
            closes_above_high,
            volume_breakout,
        ]

        if all(conditions):
            entry_price = current["close"]
            
            # Calculate base middle for stop loss
            base_middle = (recent_10["high"].max() + recent_10["low"].min()) / 2
            
            # Use SwingRiskEngine for stop loss and targets
            risk_engine = SwingRiskEngine()
            stop_loss = risk_engine.calculate_swing_stop_loss(
                strategy_group="B",
                df=df,
                entry_price=entry_price,
                base_middle=base_middle,
            )
            
            targets = risk_engine.calculate_swing_targets(df, entry_price, stop_loss)
            target = targets["target"]

            confidence = min(0.85, 0.7 + (current["volume"] / avg_volume_20 - 1) * 0.1)

            metadata = {
                "base_range_pct": float(price_range_pct * 100),
                "volume_ratio": float(current["volume"] / avg_volume_20),
                "holding_period_days": 12,
                "target_type": "breakout",
                "strategy_group": "B",
                "target_1": targets["target_1"],
                "target_2": targets["target_2"],
                "base_middle": float(base_middle),
            }

            return self.trade_assistant.create_trade_ticket(
                ticker=ticker,
                strategy_name="The Coil Breakout",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                confidence=confidence,
                metadata=metadata,
            )

        return None

