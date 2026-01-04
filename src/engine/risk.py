"""Swing Risk Engine: Strategy-group-specific stop loss and target calculation for medium-term swing trading."""

import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import pandas_ta as ta
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SwingRiskEngine:
    """
    Swing Risk Engine for calculating strategy-group-specific stop loss and targets.
    
    Designed for 10-20 day holding periods with structural setups.
    """

    def __init__(self):
        """Initialize swing risk engine."""
        self.stop_percent = getattr(config, "SWING_STOP_PERCENT", 0.02)  # 2% below support
        self.target_risk_multiplier = getattr(config, "SWING_TARGET_RISK_MULTIPLIER", 2.0)

    def get_swing_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """
        Find significant High/Low of the last N days (needed for Fib/Darvas).
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of days to look back
            
        Returns:
            Dictionary with 'swing_high' and 'swing_low'
        """
        if df.empty or len(df) < lookback:
            lookback = len(df)
        
        recent_data = df.tail(lookback)
        swing_high = float(recent_data["high"].max())
        swing_low = float(recent_data["low"].min())
        
        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "swing_high_date": recent_data["high"].idxmax(),
            "swing_low_date": recent_data["low"].idxmin(),
        }

    def calculate_swing_stop_loss(
        self,
        strategy_group: str,
        df: pd.DataFrame,
        entry_price: float,
        support_level: Optional[float] = None,
        base_middle: Optional[float] = None,
        swing_low: Optional[float] = None,
    ) -> float:
        """
        Calculate stop loss based on strategy group.
        
        Args:
            strategy_group: "A" (Support), "B" (Breakout), or "C" (Momentum)
            df: DataFrame with OHLCV data
            entry_price: Entry price
            support_level: Support level for Group A (50 EMA or Fib level)
            base_middle: Middle of base/box for Group B
            swing_low: Recent swing low for Group C
            
        Returns:
            Stop loss price
        """
        if strategy_group == "A":  # Support strategies
            if support_level is None:
                # Fallback: 2% below entry
                return entry_price * (1 - self.stop_percent)
            # 2% below support level
            stop_loss = support_level * (1 - self.stop_percent)
            # Ensure stop is below entry
            return min(stop_loss, entry_price * 0.98)
        
        elif strategy_group == "B":  # Breakout strategies
            if base_middle is None:
                # Fallback: Calculate from recent range
                recent_low = df["low"].tail(10).min()
                base_middle = (entry_price + recent_low) / 2
            # Below middle of box/base
            return base_middle * 0.99
        
        elif strategy_group == "C":  # Momentum strategies
            if swing_low is None:
                swing_levels = self.get_swing_levels(df, lookback=20)
                swing_low = swing_levels["swing_low"]
            # Below recent swing low
            return swing_low * 0.99
        
        # Default fallback
        return entry_price * 0.95

    def calculate_swing_targets(
        self,
        df: pd.DataFrame,
        entry_price: float,
        stop_loss: float,
    ) -> Dict[str, float]:
        """
        Calculate dual targets for swing trading.
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Dictionary with 'target_1' (Swing High) and 'target_2' (2.0x Risk)
        """
        # Target 1: Previous Swing High (Sell 50%)
        swing_levels = self.get_swing_levels(df, lookback=50)
        target_1 = swing_levels["swing_high"]
        
        # Ensure target 1 is above entry
        if target_1 <= entry_price:
            # Use recent high if swing high is too low
            recent_high = df["high"].tail(20).max()
            target_1 = max(recent_high, entry_price * 1.05)
        
        # Target 2: 2.0x Risk (Sell 50%)
        risk_per_share = abs(entry_price - stop_loss)
        target_2 = entry_price + (risk_per_share * self.target_risk_multiplier)
        
        # Use the higher target as primary
        primary_target = max(target_1, target_2)
        
        return {
            "target_1": target_1,
            "target_2": target_2,
            "target": primary_target,  # Primary target for TradeTicket
            "swing_high": swing_levels["swing_high"],
        }

    def calculate_fibonacci_levels(
        self,
        swing_low: float,
        swing_high: float,
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            swing_low: Swing low price
            swing_high: Swing high price
            
        Returns:
            Dictionary with fib levels (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
        """
        price_range = swing_high - swing_low
        
        return {
            "fib_0": swing_low,
            "fib_0_236": swing_low + (price_range * 0.236),
            "fib_0_382": swing_low + (price_range * 0.382),
            "fib_0_5": swing_low + (price_range * 0.5),
            "fib_0_618": swing_low + (price_range * 0.618),  # Golden Pocket
            "fib_0_786": swing_low + (price_range * 0.786),
            "fib_1": swing_high,
        }

