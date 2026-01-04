"""Pydantic schemas for Trade Ticket and trading objects."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TradeTicket(BaseModel):
    """Complete trade ticket with position sizing and risk management."""

    ticker: str = Field(..., description="Stock ticker symbol")
    strategy_name: str = Field(..., description="Strategy generating the signal")
    action: str = Field(..., description="BUY or WAIT")
    timestamp: datetime = Field(default_factory=datetime.now, description="Ticket generation time")
    entry_price: float = Field(..., gt=0, description="Entry price per share")
    stop_loss: float = Field(..., gt=0, description="Stop loss price per share")
    target: float = Field(..., gt=0, description="Target price per share")
    quantity: int = Field(..., ge=1, description="Recommended quantity of shares")
    risk_per_trade: float = Field(..., ge=0.0, le=1.0, description="Risk percentage per trade")
    risk_amount: float = Field(..., ge=0, description="Total risk amount in currency")
    reward_amount: float = Field(..., ge=0, description="Total reward amount in currency")
    risk_reward_ratio: float = Field(..., description="Risk:Reward ratio (e.g., 1:2 = 2.0)")
    account_equity: float = Field(..., gt=0, description="Account equity used for calculation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence score")
    metadata: dict = Field(default_factory=dict, description="Additional strategy-specific data")

    @field_validator("stop_loss")
    @classmethod
    def validate_stop_loss(cls, v: float, info) -> float:
        """Validate stop loss is below entry price."""
        if "entry_price" in info.data and v >= info.data["entry_price"]:
            raise ValueError("Stop loss must be below entry price")
        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: float, info) -> float:
        """Validate target is above entry price."""
        if "entry_price" in info.data and v <= info.data["entry_price"]:
            raise ValueError("Target must be above entry price")
        return v

    @field_validator("risk_reward_ratio")
    @classmethod
    def validate_risk_reward(cls, v: float) -> float:
        """Validate risk-reward ratio is positive."""
        if v <= 0:
            raise ValueError("Risk-reward ratio must be positive")
        return v

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class TradeSignal(BaseModel):
    """Ranked trade signal with quality score and holding period."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(default="", description="Company name")
    strategy_name: str = Field(..., description="Strategy generating the signal")
    entry_price: float = Field(..., gt=0, description="Entry price per share")
    stop_loss: float = Field(..., gt=0, description="Stop loss price per share")
    target: float = Field(..., gt=0, description="Target price per share")
    quantity: int = Field(..., ge=1, description="Recommended quantity of shares")
    risk_reward_ratio: float = Field(..., description="Risk:Reward ratio")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Quality score (0-100)")
    holding_period_days: int = Field(..., ge=1, description="Estimated holding period in days")
    adx_value: float = Field(..., ge=0.0, description="ADX value for trend strength")
    rsi_value: float = Field(..., ge=0.0, le=100.0, description="RSI value")
    timestamp: datetime = Field(default_factory=datetime.now, description="Signal generation time")
    metadata: dict = Field(default_factory=dict, description="Additional strategy-specific data")

    @field_validator("stop_loss")
    @classmethod
    def validate_stop_loss(cls, v: float, info) -> float:
        """Validate stop loss is below entry price."""
        if "entry_price" in info.data and v >= info.data["entry_price"]:
            raise ValueError("Stop loss must be below entry price")
        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: float, info) -> float:
        """Validate target is above entry price."""
        if "entry_price" in info.data and v <= info.data["entry_price"]:
            raise ValueError("Target must be above entry price")
        return v

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}
