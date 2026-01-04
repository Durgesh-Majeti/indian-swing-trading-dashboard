"""Strategy Registry: Manages strategy registration and loading."""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import importlib
import config
from src.engine.strategy_base import BaseStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Strategy Registry for managing plug-and-play strategies.
    
    Loads strategies from configuration and provides access to enabled strategies.
    """

    def __init__(self):
        """Initialize strategy registry."""
        self._strategies: Dict[str, BaseStrategy] = {}
        self._config = getattr(config, "STRATEGY_CONFIG", {})
        self._dynamic_config: Dict = {}  # For runtime config updates

    def register(self, strategy: BaseStrategy, name: Optional[str] = None) -> None:
        """
        Register a strategy instance.
        
        Args:
            strategy: Strategy instance inheriting from BaseStrategy
            name: Optional strategy name (defaults to strategy.get_name())
        """
        strategy_name = name or strategy.get_name().lower().replace(" ", "_")
        self._strategies[strategy_name] = strategy
        logger.info(f"Registered strategy: {strategy_name}")

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Get a strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy instance or None if not found
        """
        return self._strategies.get(name)

    def get_enabled_strategies(self, regime: str = "AGGRESSIVE") -> List[BaseStrategy]:
        """
        Get all enabled strategies that are allowed in the given regime.
        
        Args:
            regime: Market regime ("AGGRESSIVE", "DEFENSIVE", "CASH_PROTECTION")
            
        Returns:
            List of enabled strategy instances
        """
        enabled = []
        
        for name, strategy in self._strategies.items():
            # Get strategy config (merge dynamic updates with base config)
            base_config = self._config.get(name, {})
            dynamic_config = self._dynamic_config.get(name, {})
            strategy_config = {**base_config, **dynamic_config}
            
            # Check if strategy is enabled
            if not strategy_config.get("enabled", True):
                continue
            
            # Check if strategy is allowed in this regime
            allowed_regimes = strategy_config.get("allowed_regimes", ["AGGRESSIVE", "DEFENSIVE", "CASH_PROTECTION"])
            if regime not in allowed_regimes:
                continue
            
            # Check strategy's own regime check
            if not strategy.is_allowed_in_regime(regime):
                continue
            
            enabled.append(strategy)
        
        logger.debug(f"Found {len(enabled)} enabled strategies for regime {regime}")
        return enabled

    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """
        Get all registered strategies.
        
        Returns:
            Dictionary of strategy name -> strategy instance
        """
        return self._strategies.copy()

    def load_strategies_from_config(self, trade_assistant, data_engine=None) -> None:
        """
        Load strategies from configuration.
        
        This method imports and registers all strategies defined in the config.
        
        Args:
            trade_assistant: TradeAssistant instance
            data_engine: Optional FastDataEngine instance
        """
        # Import built-in swing trading strategies
        try:
            from src.engine.strategy_modules.institutional_wave import InstitutionalWaveStrategy
            from src.engine.strategy_modules.golden_pocket import GoldenPocketStrategy
            from src.engine.strategy_modules.coil_breakout import CoilBreakoutStrategy
            from src.engine.strategy_modules.darvas_box import DarvasBoxStrategy
            from src.engine.strategy_modules.weekly_climber import WeeklyClimberStrategy
            from src.engine.strategy_modules.macd_zero_turn import MACDZeroTurnStrategy
            
            # Register built-in swing strategies
            self.register(InstitutionalWaveStrategy(trade_assistant, data_engine), "institutional_wave")
            self.register(GoldenPocketStrategy(trade_assistant, data_engine), "golden_pocket")
            self.register(CoilBreakoutStrategy(trade_assistant, data_engine), "coil_breakout")
            self.register(DarvasBoxStrategy(trade_assistant, data_engine), "darvas_box")
            self.register(WeeklyClimberStrategy(trade_assistant, data_engine), "weekly_climber")
            self.register(MACDZeroTurnStrategy(trade_assistant, data_engine), "macd_zero_turn")
            
            logger.info(f"Loaded {len(self._strategies)} built-in strategies")
        except ImportError as e:
            logger.warning(f"Could not load some strategies: {e}")

    def get_strategy_config(self, name: str) -> Dict:
        """
        Get configuration for a strategy (merged with dynamic updates).
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy configuration dictionary
        """
        base_config = self._config.get(name, {}).copy()
        dynamic_config = self._dynamic_config.get(name, {})
        
        # Deep merge params
        if "params" in base_config and "params" in dynamic_config:
            base_config["params"] = {**base_config["params"], **dynamic_config["params"]}
        
        # Merge other fields
        base_config.update({k: v for k, v in dynamic_config.items() if k != "params"})
        
        return base_config

    def update_strategy_config(self, name: str, updates: Dict) -> None:
        """
        Update strategy configuration dynamically.
        
        Args:
            name: Strategy name
            updates: Dictionary of config updates
        """
        if name not in self._dynamic_config:
            self._dynamic_config[name] = {}
        
        # Deep merge params
        if "params" in updates and "params" in self._dynamic_config[name]:
            self._dynamic_config[name]["params"].update(updates["params"])
        elif "params" in updates:
            self._dynamic_config[name]["params"] = updates["params"].copy()
        
        # Merge other fields
        self._dynamic_config[name].update({k: v for k, v in updates.items() if k != "params"})
        
        logger.info(f"Updated config for strategy: {name}")

