"""Trade Assistant: Risk management, position sizing, and quality scoring."""

import logging
from typing import Optional
import pandas as pd
import pandas_ta as ta
from src.models.schemas import TradeTicket, TradeSignal
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeAssistant:
    """Trade Assistant for risk management and position sizing."""

    def __init__(self, account_equity: float = config.ACCOUNT_EQUITY, data_engine=None):
        """
        Initialize trade assistant.

        Args:
            account_equity: Total account equity
            data_engine: Optional FastDataEngine for fetching company names
        """
        self.account_equity = account_equity
        self.data_engine = data_engine

    def calculate_stop_loss(
        self, df: pd.DataFrame, entry_price: float, atr_period: int = config.ATR_PERIOD
    ) -> float:
        """
        Calculate dynamic stop loss: Low - (2.0 * ATR).

        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price
            atr_period: ATR period

        Returns:
            Stop loss price
        """
        if df.empty or len(df) < atr_period:
            return entry_price * 0.95  # Fallback to 5% stop
        
        atr = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
        if atr is None or (hasattr(atr, 'empty') and atr.empty) or len(atr) == 0:
            return entry_price * 0.95  # Fallback to 5% stop
        
        try:
            current_atr_value = atr.iloc[-1]
            if pd.isna(current_atr_value):
                return entry_price * 0.95  # Fallback to 5% stop
        except (IndexError, AttributeError):
            return entry_price * 0.95  # Fallback to 5% stop
        
        current_low = df["low"].iloc[-1]
        current_atr = float(current_atr_value)
        stop_loss = current_low - (current_atr * config.ATR_MULTIPLIER)
        
        # Ensure stop loss is reasonable (not more than 10% below entry)
        max_stop_distance = entry_price * 0.10
        min_stop_loss = entry_price - max_stop_distance
        return max(stop_loss, min_stop_loss)

    def calculate_dynamic_target(
        self,
        df: pd.DataFrame,
        entry_price: float,
        strategy_name: str,
        atr_period: int = config.ATR_PERIOD,
    ) -> float:
        """
        Calculate dynamic profit target based on strategy type.
        Strategy-specific logic ensures targets are realistically achievable.

        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price
            strategy_name: Strategy name to determine target logic
            atr_period: ATR period for volatility-based targets

        Returns:
            Dynamic target price

        Strategy-Specific Logic:
        - Pullback (Strategy A): Target = Recent Swing High (Max High of last 20 days)
          Logic: Price usually retests the previous high after a pullback.
        
        - Breakout (Strategy B, E): Target = Entry + (2.0 * ATR)
          Logic: Breakouts usually expand by 2x average volatility.
        
        - Reversion (Strategy C): Target = 50 EMA
          Logic: Price snaps back to the mean after oversold condition.
        
        - Trend Flow (Strategy D): Target = Entry + (3.0 * ATR)
          Logic: Aim for a larger extended move in strong trends.
        """
        if df.empty or len(df) < 20:
            # Fallback to fixed ratio if insufficient data
            logger.warning(f"Insufficient data for dynamic target, using fixed ratio")
            return entry_price * 1.05  # 5% default target

        # Calculate ATR for volatility-based targets
        atr = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
        if atr is None or (hasattr(atr, 'empty') and atr.empty) or len(atr) == 0:
            current_atr = entry_price * 0.02  # Fallback: 2% of entry price
        else:
            try:
                atr_value = atr.iloc[-1]
                if pd.isna(atr_value):
                    current_atr = entry_price * 0.02  # Fallback: 2% of entry price
                else:
                    current_atr = float(atr_value)
            except (IndexError, AttributeError):
                current_atr = entry_price * 0.02  # Fallback: 2% of entry price

        # Current swing trading strategies use SwingRiskEngine for targets
        # This method is kept for backward compatibility but strategies should use SwingRiskEngine
        # Default: Use ATR-based target for all strategies
        if "Institutional Wave" in strategy_name or "Golden Pocket" in strategy_name:
            # Support strategies: Target = Recent Swing High or Entry + 2*ATR
            swing_high_period = 20
            if len(df) >= swing_high_period:
                recent_high = df["high"].iloc[-swing_high_period:].max()
                target = max(recent_high, entry_price + (2.0 * current_atr))
                logger.debug(f"Support strategy: Target = {target:.2f}")
            else:
                target = entry_price + (2.0 * current_atr)
                logger.debug(f"Support strategy: Using ATR fallback = {target:.2f}")
        elif "Coil Breakout" in strategy_name or "Darvas Box" in strategy_name:
            # Breakout strategies: Target = Entry + 2*ATR
            target = entry_price + (2.0 * current_atr)
            logger.debug(f"Breakout strategy: Target = Entry + 2*ATR = {target:.2f}")
        elif "Weekly Climber" in strategy_name or "MACD Zero" in strategy_name:
            # Momentum strategies: Target = Entry + 2.5*ATR
            target = entry_price + (2.5 * current_atr)
            logger.debug(f"Momentum strategy: Target = Entry + 2.5*ATR = {target:.2f}")
        else:
            logger.warning(f"Unknown strategy '{strategy_name}', using fixed ratio target")
            target = entry_price * 1.05  # 5% default

        # Safety check: Ensure target is above entry
        if target <= entry_price:
            target = entry_price * 1.03  # Minimum 3% target
            logger.warning(f"Target was below entry, adjusted to {target:.2f}")

        return round(target, 2)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_per_trade: float = config.RISK_PER_TRADE,
        account_equity: float | None = None,
    ) -> int:
        """
        Calculate position size: Quantity = (Total_Capital * Risk_Per_Trade_%) / (Entry - Stop_Loss).

        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price per share
            risk_per_trade: Risk percentage per trade (e.g., 0.01 for 1%)
            account_equity: Account equity (uses instance default if None)

        Returns:
            Quantity of shares to buy
        """
        equity = account_equity or self.account_equity
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        risk_amount = equity * risk_per_trade
        quantity = int(risk_amount / risk_per_share)
        return max(1, quantity)

    def create_trade_ticket(
        self,
        ticker: str,
        strategy_name: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        risk_per_trade: float = config.RISK_PER_TRADE,
        account_equity: float | None = None,
        metadata: dict | None = None,
    ) -> TradeTicket:
        """
        Create a complete trade ticket with all calculations.

        Args:
            ticker: Stock ticker
            strategy_name: Strategy name
            entry_price: Entry price
            stop_loss: Stop loss price
            target: Target price
            confidence: Signal confidence
            risk_per_trade: Risk per trade percentage
            account_equity: Account equity
            metadata: Additional metadata

        Returns:
            TradeTicket object
        """
        equity = account_equity or self.account_equity
        quantity = self.calculate_position_size(entry_price, stop_loss, risk_per_trade, equity)
        
        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = abs(target - entry_price)
        
        risk_amount = risk_per_share * quantity
        reward_amount = reward_per_share * quantity
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
        
        return TradeTicket(
            ticker=ticker,
            strategy_name=strategy_name,
            action="BUY",
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            risk_per_trade=risk_per_trade,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio,
            account_equity=equity,
            confidence=confidence,
            metadata=metadata or {},
        )

    def ticket_to_signal(
        self, ticket: TradeTicket, quality_score: float, company_name: str | None = None
    ) -> TradeSignal:
        """
        Convert TradeTicket to TradeSignal with quality score.

        Args:
            ticket: Trade ticket
            quality_score: Calculated quality score
            company_name: Optional company name (fetched if not provided and data_engine available)

        Returns:
            TradeSignal object
        """
        metadata = ticket.metadata
        
        # Fetch company name if not provided
        if not company_name and self.data_engine:
            try:
                company_name = self.data_engine.get_company_name(ticket.ticker)
            except Exception:
                company_name = ticket.ticker.replace(".NS", "")
        
        # Ensure confidence is in metadata for dashboard access
        if "confidence" not in metadata:
            metadata["confidence"] = ticket.confidence
        
        return TradeSignal(
            ticker=ticket.ticker,
            company_name=company_name or ticket.ticker.replace(".NS", ""),
            strategy_name=ticket.strategy_name,
            entry_price=ticket.entry_price,
            stop_loss=ticket.stop_loss,
            target=ticket.target,
            quantity=ticket.quantity,
            risk_reward_ratio=ticket.risk_reward_ratio,
            quality_score=quality_score,
            holding_period_days=metadata.get("holding_period_days", 10),
            adx_value=metadata.get("adx", 0.0),
            rsi_value=metadata.get("rsi", 50.0),
            metadata=metadata,
        )

