"""Base Strategy Interface: Abstract class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import pandas as pd
from src.models.schemas import TradeTicket


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must inherit from this class and implement the required methods.
    """

    def __init__(self, trade_assistant, data_engine=None):
        """
        Initialize base strategy.
        
        Args:
            trade_assistant: TradeAssistant instance for creating tickets
            data_engine: Optional FastDataEngine for data access
        """
        self.trade_assistant = trade_assistant
        self.data_engine = data_engine

    @abstractmethod
    def scan(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime: str = "AGGRESSIVE",
        params: Optional[Dict] = None,
    ) -> Optional[TradeTicket]:
        """
        Scan ticker for trading opportunities.
        
        Args:
            ticker: Stock ticker symbol
            df: Pre-fetched DataFrame with OHLCV data
            regime: Market regime ("AGGRESSIVE", "DEFENSIVE", "CASH_PROTECTION")
            params: Strategy-specific parameters dictionary
            
        Returns:
            TradeTicket if opportunity found, None otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get strategy display name.
        
        Returns:
            Strategy name string
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get strategy description.
        
        Returns:
            Strategy description string
        """
        pass

    @abstractmethod
    def get_default_params(self) -> Dict:
        """
        Get default parameters for this strategy.
        
        Returns:
            Dictionary of default parameters
        """
        pass

    @abstractmethod
    def is_allowed_in_regime(self, regime: str) -> bool:
        """
        Check if strategy is allowed in given market regime.
        
        Args:
            regime: Market regime ("AGGRESSIVE", "DEFENSIVE", "CASH_PROTECTION")
            
        Returns:
            True if strategy can run in this regime, False otherwise
        """
        pass

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) for trend strength."""
        import pandas_ta as ta
        
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=period)
        if isinstance(adx_result, pd.DataFrame):
            adx_col = f"ADX_{period}"
            if adx_col in adx_result.columns:
                return adx_result[adx_col]
            for col in adx_result.columns:
                if "ADX" in col:
                    return adx_result[col]
        elif isinstance(adx_result, pd.Series):
            return adx_result
        return pd.Series(index=df.index, dtype=float)

    def _filter_dead_stocks(self, df: pd.DataFrame, min_adx: float = 20.0) -> bool:
        """Filter dead stocks (low ADX)."""
        if df.empty or len(df) < 20:
            return False
        
        adx = self._calculate_adx(df)
        if adx.empty or len(adx) == 0:
            return False
        
        current_adx = adx.iloc[-1]
        return not pd.isna(current_adx) and current_adx >= min_adx

    def get_swing_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Find significant High/Low of the last N days (helper for strategies).
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of days to look back
            
        Returns:
            Dictionary with swing_high, swing_low, and dates
        """
        from src.engine.risk import SwingRiskEngine
        risk_engine = SwingRiskEngine()
        return risk_engine.get_swing_levels(df, lookback)

    def get_strategy_group(self) -> str:
        """
        Get strategy group classification.
        
        Returns:
            "A" (Support), "B" (Breakout), or "C" (Momentum)
        """
        # Default implementation - subclasses should override
        return "A"

