"""Strategy modules package - Swing Trading Strategies."""

from src.engine.strategy_modules.institutional_wave import InstitutionalWaveStrategy
from src.engine.strategy_modules.golden_pocket import GoldenPocketStrategy
from src.engine.strategy_modules.coil_breakout import CoilBreakoutStrategy
from src.engine.strategy_modules.darvas_box import DarvasBoxStrategy
from src.engine.strategy_modules.weekly_climber import WeeklyClimberStrategy
from src.engine.strategy_modules.macd_zero_turn import MACDZeroTurnStrategy

__all__ = [
    "InstitutionalWaveStrategy",
    "GoldenPocketStrategy",
    "CoilBreakoutStrategy",
    "DarvasBoxStrategy",
    "WeeklyClimberStrategy",
    "MACDZeroTurnStrategy",
]
