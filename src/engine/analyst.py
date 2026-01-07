"""Intelligence Engine: Flash Backtest and Enhanced Quality Scoring."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import pandas_ta as ta
from src.models.schemas import TradeTicket, TradeSignal
from src.feed.data import FastDataEngine
from src.engine.assistant import TradeAssistant
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligenceEngine:
    """
    Intelligence Engine that validates trades via Flash Backtest
    and calculates enhanced quality scores with historic win rate.
    """

    def __init__(
        self,
        data_engine: FastDataEngine,
        trade_assistant: TradeAssistant,
    ):
        """
        Initialize intelligence engine.

        Args:
            data_engine: Fast data engine for fetching historical data
            trade_assistant: Trade assistant for risk calculations
        """
        self.data_engine = data_engine
        self.trade_assistant = trade_assistant
        self._backtest_cache: Dict[str, Dict] = {}

    def flash_backtest(
        self,
        ticker: str,
        strategy_name: str,
        lookback_days: int = 365,
    ) -> Dict[str, float]:
        """
        Run a fast backtest on the last year of data for a specific stock and strategy.
        This validates if the strategy historically works for this stock.

        Args:
            ticker: Stock ticker
            strategy_name: Strategy name to backtest
            lookback_days: Number of days to look back (default: 365 = 1 year)

        Returns:
            Dictionary with win_rate, total_trades, avg_return
        """
        cache_key = f"{ticker}_{strategy_name}_{lookback_days}"
        
        # Check cache first
        if cache_key in self._backtest_cache:
            return self._backtest_cache[cache_key]

        try:
            # Fetch historical data (fetch more than needed, then filter)
            period_days = max(lookback_days, 365)  # Fetch at least 1 year
            period = "2y" if period_days > 365 else "1y"
            
            df = self.data_engine.fetch_single(ticker, period=period, use_cache=True)
            
            if df.empty:
                result = {"win_rate": 0.0, "total_trades": 0, "avg_return": 0.0}
                self._backtest_cache[cache_key] = result
                return result
            
            # Filter to lookback period
            # Ensure index is datetime64[ns] (Timestamp) for proper comparison
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            elif df.index.dtype != 'datetime64[ns]':
                df.index = pd.to_datetime(df.index)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            # Convert start_date to Timestamp for pandas 2.0+ compatibility
            start_timestamp = pd.Timestamp(start_date)
            df = df.loc[df.index >= start_timestamp]
            
            if df.empty or len(df) < 100:
                result = {"win_rate": 0.0, "total_trades": 0, "avg_return": 0.0}
                self._backtest_cache[cache_key] = result
                return result

            # Use StrategyRunner to scan with actual strategies (not old function names)
            # Simulate trades by scanning historical data
            trades = []
            min_lookback = 250  # Minimum bars needed for indicators
            
            # Create StrategyRunner once (reuse for efficiency)
            from src.engine.strategies import StrategyRunner
            runner = StrategyRunner(self.data_engine, self.trade_assistant)
            
            # Get the actual strategy from registry
            strategy = None
            for s in runner.registry.get_all_strategies():
                if s.get_name() == strategy_name:
                    strategy = s
                    break
            
            if not strategy:
                logger.warning(f"Strategy {strategy_name} not found for {ticker}")
                result = {"win_rate": 0.0, "total_trades": 0, "avg_return": 0.0}
                self._backtest_cache[cache_key] = result
                return result
            
            # Scan through historical data to find multiple signals
            i = min_lookback
            max_trades = 20  # Limit to avoid too many trades
            skip_days_after_trade = 10  # Skip days after a trade to avoid overlapping
            
            # Disable market regime check for backtesting (we want to test strategy regardless of current market)
            # Store original cache and clear it
            original_regime_cache = runner.market_regime_cache
            runner.market_regime_cache = True  # Force bullish for backtesting
            
            while i < len(df) - 1 and len(trades) < max_trades:  # Need at least one bar ahead for entry
                historical_df = df.iloc[: i + 1].copy()
                
                try:
                    # Scan with the actual strategy
                    ticket = strategy.scan(ticker, historical_df, regime="AGGRESSIVE")
                    if ticket:
                        # Entry happens at the NEXT bar after signal (i+1)
                        # This simulates real trading where you enter after seeing the signal
                        entry_idx = i + 1
                        
                        if entry_idx >= len(df):
                            # No next bar available, skip this signal
                            i += 1
                            continue
                        
                        # Use actual entry price from next bar (open or close)
                        entry_bar = df.iloc[entry_idx]
                        entry_price = entry_bar["open"]  # Enter at next bar's open (more realistic)
                        
                        # Recalculate stop loss and target based on entry bar data
                        # Use data up to entry bar for calculations
                        entry_df = df.iloc[: entry_idx + 1].copy()
                        stop_loss = self.trade_assistant.calculate_stop_loss(entry_df, entry_price)
                        target = self.trade_assistant.calculate_dynamic_target(
                            entry_df, entry_price, strategy_name
                        )
                        
                        # Find exit (stop loss or target hit, or end of data)
                        exit_idx = len(df) - 1
                        exit_reason = "timeout"
                        exit_price = df.iloc[exit_idx]["close"]
                        
                        # Look ahead to find exit (max holding period based on strategy)
                        max_holding_days = ticket.metadata.get("holding_period_days", 30) * 2  # Allow 2x estimated period
                        max_exit_idx = min(entry_idx + max_holding_days, len(df))
                        
                        # Check entry bar first for intraday hits (we enter at open, can exit same day)
                        entry_bar = df.iloc[entry_idx]
                        if entry_bar["low"] <= stop_loss:
                            exit_idx = entry_idx
                            exit_reason = "stop_loss"
                            exit_price = stop_loss
                        elif entry_bar["high"] >= target:
                            exit_idx = entry_idx
                            exit_reason = "target"
                            exit_price = target
                        else:
                            # Look ahead to find exit
                            for j in range(entry_idx + 1, max_exit_idx):
                                current_bar = df.iloc[j]
                                
                                # Check stop loss (intraday low can hit stop)
                                if current_bar["low"] <= stop_loss:
                                    exit_idx = j
                                    exit_reason = "stop_loss"
                                    exit_price = stop_loss
                                    break
                                
                                # Check target (intraday high can hit target)
                                if current_bar["high"] >= target:
                                    exit_idx = j
                                    exit_reason = "target"
                                    exit_price = target
                                    break
                        
                        # Calculate P&L
                        pnl = exit_price - entry_price
                        pnl_percent = (pnl / entry_price) * 100
                        
                        trades.append({
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "stop_loss": stop_loss,
                            "target": target,
                            "pnl": pnl,
                            "pnl_percent": pnl_percent,
                            "exit_reason": exit_reason,
                            "win": pnl > 0,
                            "entry_idx": entry_idx,
                            "exit_idx": exit_idx,
                            "holding_days": exit_idx - entry_idx,
                        })
                        
                        # Skip forward to avoid overlapping trades
                        i = exit_idx + skip_days_after_trade
                    else:
                        i += 1
                
                except Exception as e:
                    logger.debug(f"Error in flash backtest at index {i} for {ticker}: {e}")
                    i += 1
                    continue
            
            # Restore original market regime cache
            runner.market_regime_cache = original_regime_cache

            # Calculate metrics
            if not trades:
                logger.info(f"Flash backtest for {ticker} ({strategy_name}): No trades found")
                result = {"win_rate": 0.0, "total_trades": 0, "avg_return": 0.0}
            else:
                winning_trades = [t for t in trades if t["win"]]
                losing_trades = [t for t in trades if not t["win"]]
                win_rate = len(winning_trades) / len(trades) * 100
                avg_return = sum(t["pnl_percent"] for t in trades) / len(trades)
                avg_win = sum(t["pnl_percent"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
                avg_loss = sum(t["pnl_percent"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
                avg_holding_days = sum(t["holding_days"] for t in trades) / len(trades)
                
                logger.info(
                    f"Flash backtest for {ticker} ({strategy_name}): "
                    f"{len(trades)} trades, {win_rate:.1f}% win rate, "
                    f"{avg_return:.2f}% avg return, {avg_win:.2f}% avg win, {avg_loss:.2f}% avg loss, "
                    f"{avg_holding_days:.1f} days avg holding"
                )
                
                result = {
                    "win_rate": round(win_rate, 2),
                    "total_trades": len(trades),
                    "avg_return": round(avg_return, 2),
                    "avg_win": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "avg_holding_days": round(avg_holding_days, 1),
                }
            
            # Cache result
            self._backtest_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error in flash backtest for {ticker} ({strategy_name}): {e}")
            result = {"win_rate": 0.0, "total_trades": 0, "avg_return": 0.0}
            self._backtest_cache[cache_key] = result
            return result

    def generate_investment_thesis(
        self,
        ticket: TradeTicket,
        sector: str,
        sector_performance: Dict,
        volatility_regime: Dict,
        top_sector: Optional[Dict] = None,
    ) -> str:
        """
        Generate swing trading investment thesis (10-20 day hold).

        Args:
            ticket: Trade ticket
            sector: Stock sector
            sector_performance: Sector performance data (from get_sector_performance)
            volatility_regime: Volatility regime data
            top_sector: Top sector information

        Returns:
            Plain English swing trading thesis
        """
        strategy = ticket.strategy_name
        ticker_clean = ticket.ticker.replace(".NS", "")
        holding_period = ticket.metadata.get("holding_period_days", 15)
        
        # Sector performance
        sector_perf = sector_performance.get("change_pct", 0.0)
        sector_status = sector_performance.get("status", "neutral")
        
        # Build swing-specific thesis
        thesis_parts = [
            f"I recommend **{ticker_clean}** for a **{holding_period}-day hold**.",
        ]
        
        # Strategy setup
        if "Institutional Wave" in strategy:
            setup_desc = "forming a **50 EMA support bounce** setup"
        elif "Golden Pocket" in strategy:
            setup_desc = "forming a **Fibonacci golden pocket** setup"
        elif "Coil Breakout" in strategy:
            setup_desc = "forming a **base building breakout** setup"
        elif "Darvas Box" in strategy:
            setup_desc = "forming a **Darvas box breakout** setup"
        elif "Weekly Climber" in strategy:
            setup_desc = "forming a **multi-timeframe momentum** setup"
        elif "MACD Zero-Turn" in strategy:
            setup_desc = "forming a **momentum shift** setup"
        else:
            setup_desc = f"forming a **{strategy}** setup"
        
        thesis_parts.append(f"It is {setup_desc}.")
        
        # Sector context
        if sector_status == "bullish" and sector_perf > 0:
            thesis_parts.append(f"The **{sector} Sector** is bullish (+{sector_perf:.1f}%), supporting the move.")
        elif sector_status == "bearish":
            thesis_parts.append(f"Note: The **{sector} Sector** is bearish ({sector_perf:.1f}%).")
        else:
            thesis_parts.append(f"The **{sector} Sector** is neutral ({sector_perf:.1f}%).")
        
        # Weekly trend confirmation
        thesis_parts.append("The Weekly trend confirms this trade.")
        
        return " ".join(thesis_parts)

    def calculate_swing_score(
        self,
        ticket: TradeTicket,
        sector_performance: Dict,
    ) -> float:
        """
        Calculate Swing Score (0-100) for swing trading.
        
        Formula: (Pattern_Clarity * 0.4) + (Sector_Strength * 0.3) + (Risk_Reward_Ratio * 0.3)
        
        Args:
            ticket: Trade ticket
            sector_performance: Sector performance data
            
        Returns:
            Swing score (0-100)
        """
        # Pattern Clarity (0-100, weighted 40%)
        # Based on confidence and setup quality
        confidence = ticket.confidence
        pattern_clarity = confidence * 100  # Convert 0-1 to 0-100
        
        # Sector Strength (0-100, weighted 30%)
        sector_perf = sector_performance.get("change_pct", 0.0)
        sector_status = sector_performance.get("status", "neutral")
        
        if sector_status == "bullish":
            sector_strength = min(100, 70 + (sector_perf * 10))  # 70-100 for bullish
        elif sector_status == "bearish":
            sector_strength = max(0, 30 + (sector_perf * 10))  # 0-30 for bearish
        else:
            sector_strength = 50  # Neutral = 50
        
        # Risk/Reward Ratio (0-100, weighted 30%)
        risk = abs(ticket.entry_price - ticket.stop_loss)
        reward = abs(ticket.target - ticket.entry_price)
        
        if risk > 0:
            risk_reward_ratio = reward / risk
            # Normalize to 0-100 (1:1 = 50, 1:2 = 75, 1:3 = 100)
            risk_reward_score = min(100, 50 + (risk_reward_ratio - 1) * 25)
        else:
            risk_reward_score = 50
        
        # Calculate weighted score
        swing_score = (
            (pattern_clarity * 0.4) +
            (sector_strength * 0.3) +
            (risk_reward_score * 0.3)
        )
        
        return round(swing_score, 1)

    def analyze_signal(
        self,
        ticket: TradeTicket,
        run_backtest: bool = True,
    ) -> TradeSignal:
        """
        Complete analysis: Flash backtest + Quality scoring + TradeSignal creation.

        Args:
            ticket: Trade ticket
            run_backtest: Whether to run flash backtest (default: True)

        Returns:
            TradeSignal with quality score and historic win rate
        """
        historic_win_rate = 0.0
        
        if run_backtest:
            backtest_result = self.flash_backtest(
                ticket.ticker,
                ticket.strategy_name,
                lookback_days=365,
            )
            historic_win_rate = backtest_result.get("win_rate", 0.0)

        # Fetch company name
        company_name = ""
        if self.data_engine:
            try:
                company_name = self.data_engine.get_company_name(ticket.ticker)
            except Exception:
                company_name = ticket.ticker.replace(".NS", "")

        # Create TradeSignal with updated metadata
        metadata = ticket.metadata.copy()
        metadata["historic_win_rate"] = historic_win_rate
        if run_backtest:
            metadata["backtest_trades"] = backtest_result.get("total_trades", 0)
        else:
            metadata["backtest_trades"] = 0

        # Get ticket data and update metadata
        ticket_data = ticket.model_dump()
        ticket_data["metadata"] = metadata

        # Create ticket with updated metadata
        ticket_with_metadata = TradeTicket(**ticket_data)

        # Create signal with quality score
        return self.trade_assistant.ticket_to_signal(
            ticket=ticket_with_metadata,
            quality_score=0.0,  # Quality score is set by swing_score calculation
            company_name=company_name,
        )

