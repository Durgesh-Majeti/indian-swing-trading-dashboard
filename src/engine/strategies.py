"""Multi-Strategy Engine: Plug-and-Play Strategy Runner."""

import logging
from typing import List, Optional, Dict
import pandas as pd
from src.models.schemas import TradeTicket
from src.feed.data import FastDataEngine
from src.engine.assistant import TradeAssistant
from src.engine.strategy_registry import StrategyRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Simplified Strategy Runner using plug-and-play strategy registry.
    
    Strategies are loaded from the registry and can be enabled/disabled via configuration.
    """

    def __init__(
        self,
        data_engine: FastDataEngine,
        trade_assistant: TradeAssistant,
    ):
        """
        Initialize strategy runner.

        Args:
            data_engine: Fast data engine instance
            trade_assistant: Trade assistant instance
        """
        self.data_engine = data_engine
        self.trade_assistant = trade_assistant
        
        # Initialize strategy registry
        self.registry = StrategyRegistry()
        self.registry.load_strategies_from_config(trade_assistant, data_engine)

    def scan_ticker(
        self, ticker: str, df: pd.DataFrame, regime: str = "AGGRESSIVE", params: Optional[Dict] = None
    ) -> List[TradeTicket]:
        """
        Scan a single ticker with all enabled strategies.

        Args:
            ticker: Stock ticker
            df: Pre-fetched DataFrame
            regime: Market regime
            params: Strategy parameters (optional, overrides config)

        Returns:
            List of trade tickets
        """
        tickets: List[TradeTicket] = []

        # Get enabled strategies for this regime
        enabled_strategies = self.registry.get_enabled_strategies(regime)

        # Try each enabled strategy
        for strategy in enabled_strategies:
            try:
                # Get strategy config - use registry key format (remove "the_" prefix if present, convert hyphens to underscores)
                strategy_display_name = strategy.get_name().lower().replace(" ", "_").replace("-", "_")
                # Remove "the_" prefix to match config keys (e.g., "the_institutional_wave" -> "institutional_wave")
                strategy_name = strategy_display_name.replace("the_", "") if strategy_display_name.startswith("the_") else strategy_display_name
                strategy_config = self.registry.get_strategy_config(strategy_name)
                
                # Merge config params with provided params
                strategy_params = strategy_config.get("params", {})
                if params:
                    strategy_params.update(params)
                
                # Scan with strategy
                ticket = strategy.scan(ticker, df, regime=regime, params=strategy_params)
                
                if ticket:
                    tickets.append(ticket)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.get_name()} for {ticker}: {e}")

        return tickets

    def scan_batch(
        self, ticker_data: Dict[str, pd.DataFrame], regime: str = "AGGRESSIVE", params: Optional[Dict] = None
    ) -> List[TradeTicket]:
        """
        Scan multiple tickers with all enabled strategies.

        Args:
            ticker_data: Dictionary mapping ticker to DataFrame
            regime: Market regime
            params: Strategy parameters

        Returns:
            List of all trade tickets found
        """
        all_tickets: List[TradeTicket] = []

        for ticker, df in ticker_data.items():
            tickets = self.scan_ticker(ticker, df, regime=regime, params=params)
            all_tickets.extend(tickets)

        logger.info(f"Found {len(all_tickets)} signals across all strategies")
        return all_tickets

    def get_registry(self) -> StrategyRegistry:
        """
        Get the strategy registry.
        
        Returns:
            StrategyRegistry instance
        """
        return self.registry
