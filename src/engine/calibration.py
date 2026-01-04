"""Calibration Engine: Adaptive Parameter Adjustment (Thermostat Loop)."""

import logging
from typing import List, Dict, Optional
from src.models.schemas import TradeTicket
from src.engine.strategies import StrategyRunner
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationEngine:
    """
    Thermostat Loop: Adaptively adjusts strategy parameters to find opportunities.
    
    Logic:
    1. Run 1 (Sniper Mode): Strict params
    2. If < 5 opportunities: Run 2 (Rifle Mode): Relaxed params
    3. If still low: Run 3 (Shotgun Mode): Loose params
    """

    def __init__(self, strategy_runner: StrategyRunner):
        """
        Initialize calibration engine.

        Args:
            strategy_runner: Strategy runner instance
        """
        self.strategy_runner = strategy_runner

    def calibrate_and_scan(
        self,
        ticker_data: Dict[str, any],
        regime: str,
        min_opportunities: int = 5,
    ) -> List[TradeTicket]:
        """
        Run calibration loop with adaptive parameters.

        Args:
            ticker_data: Dictionary of ticker -> DataFrame
            regime: Market regime ("AGGRESSIVE", "DEFENSIVE", "CASH_PROTECTION")
            min_opportunities: Minimum opportunities threshold

        Returns:
            List of trade tickets with quality tags
        """
        all_opportunities = []
        
        # Run 1: Sniper Mode (Strict)
        logger.info("🔫 Sniper Mode: Running with strict parameters...")
        params_sniper = self._get_params_for_mode("SNIPER", regime)
        opportunities_sniper = self._run_scan_with_params(ticker_data, params_sniper, regime)
        
        for ticket in opportunities_sniper:
            ticket.metadata["quality_tag"] = "HIGH"
            ticket.metadata["calibration_mode"] = "SNIPER"
        all_opportunities.extend(opportunities_sniper)
        
        logger.info(f"Sniper Mode found {len(opportunities_sniper)} opportunities")
        
        # Run 2: Rifle Mode (Relaxed) - if needed
        if len(all_opportunities) < min_opportunities:
            logger.info("🔫 Rifle Mode: Relaxing parameters...")
            params_rifle = self._get_params_for_mode("RIFLE", regime)
            opportunities_rifle = self._run_scan_with_params(ticker_data, params_rifle, regime)
            
            # Filter out duplicates (same ticker+strategy)
            existing_keys = {(t.ticker, t.strategy_name) for t in all_opportunities}
            new_opportunities = [
                t for t in opportunities_rifle
                if (t.ticker, t.strategy_name) not in existing_keys
            ]
            
            for ticket in new_opportunities:
                ticket.metadata["quality_tag"] = "STANDARD"
                ticket.metadata["calibration_mode"] = "RIFLE"
            all_opportunities.extend(new_opportunities)
            
            logger.info(f"Rifle Mode found {len(new_opportunities)} additional opportunities")
        
        # Run 3: Shotgun Mode (Loose) - if still needed
        if len(all_opportunities) < min_opportunities:
            logger.info("🔫 Shotgun Mode: Using loose parameters (High Risk)...")
            params_shotgun = self._get_params_for_mode("SHOTGUN", regime)
            opportunities_shotgun = self._run_scan_with_params(ticker_data, params_shotgun, regime)
            
            # Filter out duplicates
            existing_keys = {(t.ticker, t.strategy_name) for t in all_opportunities}
            new_opportunities = [
                t for t in opportunities_shotgun
                if (t.ticker, t.strategy_name) not in existing_keys
            ]
            
            for ticket in new_opportunities:
                ticket.metadata["quality_tag"] = "LOOSE"
                ticket.metadata["calibration_mode"] = "SHOTGUN"
            all_opportunities.extend(new_opportunities)
            
            logger.info(f"Shotgun Mode found {len(new_opportunities)} additional opportunities")
        
        logger.info(f"Total opportunities after calibration: {len(all_opportunities)}")
        return all_opportunities

    def _get_params_for_mode(self, mode: str, regime: str) -> Dict:
        """
        Get parameters for calibration mode.

        Args:
            mode: "SNIPER" | "RIFLE" | "SHOTGUN"
            regime: Market regime

        Returns:
            Parameter dictionary
        """
        if mode == "SNIPER":
            return {
                "min_adx": 30,
                "min_volume_multiplier": 1.5,
                "crsi_low": 10,
                "strict_ema": True,
            }
        elif mode == "RIFLE":
            return {
                "min_adx": 25,
                "min_volume_multiplier": 1.2,
                "crsi_low": 15,
                "strict_ema": False,
            }
        else:  # SHOTGUN
            return {
                "min_adx": 20,
                "min_volume_multiplier": 1.0,
                "crsi_low": 20,
                "strict_ema": False,
            }

    def _run_scan_with_params(
        self,
        ticker_data: Dict[str, any],
        params: Dict,
        regime: str,
    ) -> List[TradeTicket]:
        """
        Run strategy scan with specific parameters.

        Args:
            ticker_data: Dictionary of ticker -> DataFrame
            params: Parameter dictionary (merged with strategy defaults)
            regime: Market regime

        Returns:
            List of trade tickets
        """
        # Pass params directly to StrategyRunner which will merge with strategy defaults
        all_tickets = self.strategy_runner.scan_batch(ticker_data, regime=regime, params=params)
        
        # Filter based on params (post-scan validation)
        # Only filter if the metadata field exists (don't filter strategies that don't set these fields)
        filtered = []
        for ticket in all_tickets:
            metadata = ticket.metadata
            
            # Check ADX if present in params AND in metadata (only filter strategies that calculate ADX)
            if "min_adx" in params and "adx" in metadata:
                adx = metadata.get("adx", 0.0)
                if adx < params["min_adx"]:
                    continue
            
            # Check volume if present in params AND in metadata (only filter strategies that calculate volume_ratio)
            if "min_volume_multiplier" in params and "volume_ratio" in metadata:
                volume_ratio = metadata.get("volume_ratio", 1.0)
                if volume_ratio < params["min_volume_multiplier"]:
                    continue
            
            filtered.append(ticket)
        
        return filtered

    def calibrate(
        self,
        tickets: List[TradeTicket],
        regime: str,
    ) -> List[TradeTicket]:
        """
        Calibration method that tags existing tickets with quality tags based on ticket characteristics.
        Uses ticket metadata to determine quality (HIGH/STANDARD/LOOSE).

        Args:
            tickets: List of already-scanned trade tickets
            regime: Market regime (for future use)

        Returns:
            List of trade tickets with quality tags
        """
        for ticket in tickets:
            if "quality_tag" in ticket.metadata:
                continue  # Already tagged
            
            metadata = ticket.metadata
            quality_score = 0
            
            # Check ADX (if available) - Higher ADX = better quality
            adx = metadata.get("adx", 0.0)
            if adx >= 30:
                quality_score += 3
            elif adx >= 25:
                quality_score += 2
            elif adx >= 20:
                quality_score += 1
            
            # Check Volume Ratio (if available) - Higher volume = better quality
            volume_ratio = metadata.get("volume_ratio", 1.0)
            if volume_ratio >= 1.5:
                quality_score += 2
            elif volume_ratio >= 1.2:
                quality_score += 1
            
            # Check Risk:Reward Ratio - Higher R:R = better quality
            rr_ratio = ticket.risk_reward_ratio
            if rr_ratio >= 3.0:
                quality_score += 2
            elif rr_ratio >= 2.0:
                quality_score += 1
            
            # Check Confidence/Pattern Clarity - Higher confidence = better quality
            confidence = metadata.get("confidence", 0.0)
            if confidence >= 0.8:
                quality_score += 2
            elif confidence >= 0.6:
                quality_score += 1
            
            # Determine quality tag based on score
            if quality_score >= 6:
                ticket.metadata["quality_tag"] = "HIGH"
                ticket.metadata["calibration_mode"] = "SNIPER"
            elif quality_score >= 3:
                ticket.metadata["quality_tag"] = "STANDARD"
                ticket.metadata["calibration_mode"] = "RIFLE"
            else:
                ticket.metadata["quality_tag"] = "LOOSE"
                ticket.metadata["calibration_mode"] = "SHOTGUN"
        
        high_count = sum(1 for t in tickets if t.metadata.get("quality_tag") == "HIGH")
        standard_count = sum(1 for t in tickets if t.metadata.get("quality_tag") == "STANDARD")
        loose_count = sum(1 for t in tickets if t.metadata.get("quality_tag") == "LOOSE")
        
        logger.info(f"Tagged {len(tickets)} tickets: HIGH={high_count}, STANDARD={standard_count}, LOOSE={loose_count}")
        return tickets

