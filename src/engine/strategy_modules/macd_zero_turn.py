"""The MACD Zero-Turn Strategy - Momentum Shift."""

from typing import Optional, Dict
import pandas as pd
import pandas_ta as ta
from src.engine.strategy_base import BaseStrategy
from src.engine.risk import SwingRiskEngine
from src.models.schemas import TradeTicket
import config


class MACDZeroTurnStrategy(BaseStrategy):
    """
    Strategy: The MACD Zero-Turn (Momentum Shift).
    
    Concept: Catching the transition from Correction to Impulse.
    Rules:
    - Setup: MACD Histogram has been Red for > 5 days
    - Trigger: MACD Histogram flips Green AND MACD Line crosses above Signal Line
    """

    def get_name(self) -> str:
        """Get strategy name."""
        return "The MACD Zero-Turn"

    def get_description(self) -> str:
        """Get strategy description."""
        return "Momentum Shift: MACD Histogram red > 5 days, flips green, MACD crosses signal"

    def get_default_params(self) -> Dict:
        """Get default parameters."""
        return {
            "histogram_days": config.MACD_ZERO_TURN_HISTOGRAM_DAYS,
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
        """Scan for MACD Zero-Turn setup."""
        # Merge with defaults to ensure all required params are present
        default_params = self.get_default_params()
        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        if df.empty or len(df) < 30:
            return None

        # Calculate MACD
        macd_result = ta.macd(df["close"])
        if macd_result is None or len(macd_result) < params["histogram_days"] + 2:
            return None

        if isinstance(macd_result, pd.DataFrame):
            macd_line = macd_result.iloc[:, 0]  # MACD line
            signal_line = macd_result.iloc[:, 1]  # Signal line
            histogram = macd_line - signal_line  # Histogram
        else:
            return None

        if histogram.empty or len(histogram) < params["histogram_days"] + 1:
            return None

        # Setup: MACD Histogram has been Red for > 5 days
        recent_histogram = histogram.tail(params["histogram_days"] + 1)
        histogram_red_days = sum(1 for h in recent_histogram.iloc[:-1] if h < 0)
        was_red = histogram_red_days >= params["histogram_days"]

        # Trigger: MACD Histogram flips Green (current > 0, previous < 0)
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        histogram_flips_green = current_hist > 0 and prev_hist < 0

        # Trigger: MACD Line crosses above Signal Line
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        macd_crosses_signal = (
            current_macd > current_signal and
            prev_macd <= prev_signal
        )

        conditions = [
            was_red,
            histogram_flips_green,
            macd_crosses_signal,
        ]

        if all(conditions):
            current = df.iloc[-1]
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

            confidence = 0.85  # High confidence for momentum shift

            metadata = {
                "macd": float(current_macd),
                "signal": float(current_signal),
                "histogram": float(current_hist),
                "histogram_red_days": histogram_red_days,
                "holding_period_days": 15,
                "target_type": "momentum",
                "strategy_group": "C",
                "target_1": targets["target_1"],
                "target_2": targets["target_2"],
            }

            return self.trade_assistant.create_trade_ticket(
                ticker=ticker,
                strategy_name="The MACD Zero-Turn",
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                confidence=confidence,
                metadata=metadata,
            )

        return None

