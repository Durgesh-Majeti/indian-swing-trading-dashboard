"""The Darvas Box Strategy - Trend Ladder."""

from typing import Optional, Dict
import pandas as pd
import pandas_ta as ta
from src.engine.strategy_base import BaseStrategy
from src.engine.risk import SwingRiskEngine
from src.models.schemas import TradeTicket
import config


class DarvasBoxStrategy(BaseStrategy):
    """
    Strategy: The Darvas Box (Trend Ladder).
    
    Concept: Resumption of trend after a pause.
    Rules:
    - Box: Identify a Resistance level tested 2+ times in 20 days but not broken
    - Trigger: Price closes > Resistance Level
    - Context: ADX > 25 (Trend exists)
    """

    def get_name(self) -> str:
        """Get strategy name."""
        return "The Darvas Box"

    def get_description(self) -> str:
        """Get strategy description."""
        return "Trend Ladder: Resistance tested 2+ times, Breakout above resistance, ADX > 25"

    def get_default_params(self) -> Dict:
        """Get default parameters."""
        return {
            "min_adx": config.DARVAS_BOX_ADX_MIN,
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
        """Scan for Darvas Box setup."""
        # Merge with defaults to ensure all required params are present
        default_params = self.get_default_params()
        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        if df.empty or len(df) < 50:
            return None

        # Context: ADX > 25 (Trend exists)
        adx = self._calculate_adx(df)
        if adx.empty or len(adx) == 0:
            return None
        
        current_adx = adx.iloc[-1]
        if pd.isna(current_adx) or current_adx < params["min_adx"]:
            return None

        current = df.iloc[-1]
        recent_20 = df.tail(20)

        # Box: Identify a Resistance level tested 2+ times in 20 days but not broken
        # Find local highs that were tested but not broken
        highs = recent_20["high"].values
        resistance_candidates = []
        
        for i in range(len(highs) - 1):
            high_value = highs[i]
            # Check if this high was tested again later (within 2% tolerance)
            tests = sum(1 for h in highs[i+1:] if abs(h - high_value) / high_value <= 0.02)
            if tests >= 1:  # Tested at least once more
                resistance_candidates.append(high_value)
        
        if not resistance_candidates:
            return None
        
        # Use the most recent resistance level
        resistance_level = max(resistance_candidates)
        
        # Ensure resistance was not broken in the last 20 days
        broken = any(close > resistance_level * 1.01 for close in recent_20["close"].values[:-1])
        if broken:
            return None

        # Trigger: Price closes > Resistance Level
        closes_above_resistance = current["close"] > resistance_level

        if closes_above_resistance:
            entry_price = current["close"]
            
            # Calculate box middle for stop loss
            box_low = recent_20["low"].min()
            box_middle = (resistance_level + box_low) / 2
            
            # Use SwingRiskEngine for stop loss and targets
            risk_engine = SwingRiskEngine()
            stop_loss = risk_engine.calculate_swing_stop_loss(
                strategy_group="B",
                df=df,
                entry_price=entry_price,
                base_middle=box_middle,
            )
            
            targets = risk_engine.calculate_swing_targets(df, entry_price, stop_loss)
            target = targets["target"]

            confidence = min(0.9, 0.7 + (current_adx / 50) * 0.2)

            metadata = {
                "resistance_level": float(resistance_level),
                "box_low": float(box_low),
                "adx": float(current_adx),
                "holding_period_days": 15,
                "target_type": "breakout",
                "strategy_group": "B",
                "target_1": targets["target_1"],
                "target_2": targets["target_2"],
                "box_middle": float(box_middle),
            }

            return self.trade_assistant.create_trade_ticket(
                ticker=ticker,
                strategy_name="The Darvas Box",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                confidence=confidence,
                metadata=metadata,
            )

        return None

