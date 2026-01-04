"""Quantitative Math Engine: Alpha Score and Kelly-Lite Position Sizing."""

import logging
from typing import Dict
import numpy as np
from src.models.schemas import TradeTicket
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantMath:
    """
    Quantitative calculations for Alpha Score and position sizing.
    """

    def calculate_alpha_score(
        self,
        ticket: TradeTicket,
        historic_win_rate: float = 0.0,
    ) -> float:
        """
        Calculate Alpha Score (0-100) for ranking trades.
        
        Formula:
        Alpha = (Trend_Strength * 0.3) + (KPI_Quality * 0.4) + (Risk_Reward * 0.3)
        
        Where:
        - Trend_Strength = ADX normalized to 0-100
        - KPI_Quality = Strategy-specific KPI (RS Ratio, OBV, VWAP, etc.) normalized
        - Risk_Reward = Risk:Reward ratio normalized to 0-100

        Args:
            ticket: Trade ticket with metadata
            historic_win_rate: Historic win rate from backtest (0-100) - used as KPI proxy

        Returns:
            Alpha Score (0-100)
        """
        metadata = ticket.metadata

        # Trend Strength component (0-100, weighted 30%)
        adx_value = metadata.get("adx", 0.0)
        trend_strength_score = min(adx_value / 50.0 * 100, 100) * 0.3

        # KPI Quality component (0-100, weighted 40%)
        # Strategy-specific KPIs
        strategy = ticket.strategy_name
        kpi_score = 0.0
        
        if "Alpha Leader" in strategy:
            rs_ratio = metadata.get("rs_ratio", 1.0)
            kpi_score = min((rs_ratio - 1.0) * 50, 100)  # 1.0 = 0, 3.0 = 100
        elif "Silent Accumulation" in strategy:
            obv_ratio = metadata.get("obv", 0.0) / max(metadata.get("obv_high", 1.0), 1.0)
            kpi_score = min(obv_ratio * 100, 100)
        elif "Institutional Floor" in strategy:
            volume_ratio = metadata.get("volume_ratio", 1.0)
            kpi_score = min((volume_ratio - 1.0) * 50, 100)
        else:
            # Use historic win rate as proxy for KPI quality
            kpi_score = min(historic_win_rate, 100)
        
        kpi_component = kpi_score * 0.4

        # Risk:Reward component (0-100, weighted 30%)
        rr_ratio = ticket.risk_reward_ratio
        # Normalize: 1:1 = 50, 1:2 = 100, 1:3 = 150 (capped at 100)
        rr_score = min(rr_ratio / 2.0 * 100, 100) * 0.3

        # Calculate total alpha score
        alpha_score = trend_strength_score + kpi_component + rr_score

        logger.debug(
            f"Alpha Score for {ticket.ticker}: "
            f"Trend={trend_strength_score:.2f}, KPI={kpi_component:.2f}, "
            f"RR={rr_score:.2f}, Total={alpha_score:.2f}"
        )

        return round(alpha_score, 2)

    def calculate_kelly_lite_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_equity: float,
        alpha_score: float,
        base_risk_percent: float = config.RISK_PER_TRADE,
    ) -> Dict[str, float]:
        """
        Calculate position size using Kelly-Lite scaling based on Alpha Score.
        
        Kelly-Lite Formula:
        - Base Risk: 1% (Alpha < 60)
        - Scaled Risk: 1% to 2% (Alpha 60-100)
        - Position Size = (Account Equity * Risk%) / (Entry - Stop Loss)

        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price per share
            account_equity: Total account equity
            alpha_score: Alpha score (0-100)
            base_risk_percent: Base risk percentage (default: 1%)

        Returns:
            Dictionary with:
            - quantity: Number of shares to buy
            - risk_amount: Total risk amount
            - risk_percent: Actual risk percentage used
        """
        # Calculate risk per share
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            logger.warning(f"Invalid risk per share: {risk_per_share}")
            return {
                "quantity": 0,
                "risk_amount": 0.0,
                "risk_percent": 0.0,
            }

        # Scale risk based on Alpha Score
        if alpha_score < 60:
            # Low confidence: Use base risk
            risk_percent = base_risk_percent
        elif alpha_score >= 90:
            # High confidence: Use maximum risk (2%)
            risk_percent = base_risk_percent * 2.0
        else:
            # Linear scaling between 60-90
            # At 60: 1%, At 90: 2%
            risk_percent = base_risk_percent * (1.0 + (alpha_score - 60) / 30.0)

        # Calculate position size
        risk_amount = account_equity * risk_percent
        quantity = int(risk_amount / risk_per_share)

        # Ensure minimum quantity of 1 if risk is positive
        if quantity < 1 and risk_amount > 0:
            quantity = 1

        logger.debug(
            f"Kelly-Lite Position Sizing for entry={entry_price:.2f}, stop={stop_loss:.2f}: "
            f"Alpha={alpha_score:.1f}, Risk%={risk_percent*100:.2f}%, "
            f"Quantity={quantity}, Risk Amount=₹{risk_amount:.2f}"
        )

        return {
            "quantity": quantity,
            "risk_amount": round(risk_amount, 2),
            "risk_percent": round(risk_percent * 100, 2),
        }

    def calculate_estimated_profit(
        self,
        entry_price: float,
        target_price: float,
        quantity: int,
    ) -> Dict[str, float]:
        """
        Calculate estimated profit and return percentage.

        Args:
            entry_price: Entry price per share
            target_price: Target price per share
            quantity: Number of shares

        Returns:
            Dictionary with:
            - profit_amount: Total profit amount
            - profit_percent: Profit percentage
        """
        profit_per_share = target_price - entry_price
        profit_amount = profit_per_share * quantity
        profit_percent = (profit_per_share / entry_price) * 100

        return {
            "profit_amount": round(profit_amount, 2),
            "profit_percent": round(profit_percent, 2),
        }

