"""
Integrated Trading System
Main orchestrator that combines wave hypothesis engine, ML ranker, trade executor, and data storage.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
import json
import asyncio
import threading

from .wave_hypothesis_engine import WaveHypothesisEngine, WaveHypothesis
from .ml_wave_ranker import MLWaveRanker
from .trade_executor import TradeExecutor, Position
from .data_storage import DataStorage
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class IntegratedTradingSystem:
    """
    Main orchestrator for the Elliott Wave trading system.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.hypothesis_engine = WaveHypothesisEngine(config)
        self.ml_ranker = MLWaveRanker()  # Would load trained model in production
        self.trade_executor = TradeExecutor(config)
        self.data_storage = DataStorage(config.get('database_path', './data/elliott_wave_data.db'))
        self.technical_indicators = TechnicalIndicators()
        
        # State tracking
        self.is_running = False
        self.last_analysis_time = None
        self.current_market_data = None
        self.current_technical_indicators = {}
        
        logger.info("IntegratedTradingSystem initialized")
        
    def analyze_market_data(self, price_data: pd.DataFrame) -> Dict:
        """
        Perform complete market analysis including wave detection, hypothesis generation, and ML ranking.
        
        Args:
            price_data: DataFrame with timestamp index and price data
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting market analysis")
        analysis_start_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Store market data
        self.current_market_data = price_data
        self.data_storage.store_market_data(price_data, "SYMBOL")  # Would be dynamic symbol
        
        # Calculate technical indicators
        self.current_technical_indicators = self.technical_indicators.calculate_all_indicators(price_data)
        
        # Generate new wave hypotheses
        new_hypothesis_ids = self.hypothesis_engine.generate_new_hypotheses(price_data)
        logger.info(f"Generated {len(new_hypothesis_ids)} new hypotheses")
        
        # Prune any invalidated hypotheses
        invalidated_ids = self.hypothesis_engine.prune_invalid_hypotheses()
        if invalidated_ids:
            logger.info(f"Pruned {len(invalidated_ids)} invalidated hypotheses")
            # Check if any open positions need to be exited
            exit_messages = self.trade_executor.check_for_invalidation_exits(invalidated_ids)
            for msg in exit_messages:
                logger.info(msg)
                
        # Get admissible hypotheses
        admissible_hypotheses = self.hypothesis_engine.get_admissible_hypotheses()
        logger.info(f"Found {len(admissible_hypotheses)} admissible hypotheses")
        
        # Rank hypotheses using ML
        if admissible_hypotheses:
            ml_scores = self.ml_ranker.rank_hypotheses(
                admissible_hypotheses, 
                price_data, 
                self.current_technical_indicators
            )
            
            # Update hypotheses with ML scores
            self.hypothesis_engine.update_hypotheses_with_ml_scores(ml_scores)
            
            # Log top ranked hypotheses
            sorted_hypotheses = sorted(admissible_hypotheses, 
                                    key=lambda h: h.ml_ranking_score, 
                                    reverse=True)
            logger.info(f"Top hypothesis score: {sorted_hypotheses[0].ml_ranking_score if sorted_hypotheses else 0.0}")
            
        # Get the best hypothesis
        best_hypothesis = self.hypothesis_engine.get_best_hypothesis()
        
        # Store active hypotheses
        for hypothesis in self.hypothesis_engine.get_active_hypotheses():
            self.data_storage.store_hypothesis(hypothesis.to_dict())
            
        # Prepare analysis result
        analysis_id = f"analysis_{analysis_start_time.strftime('%Y%m%d_%H%M%S')}"
        result = {
            'id': analysis_id,
            'timestamp': analysis_start_time.isoformat(),
            'symbol': "SYMBOL",  # Would be dynamic
            'wave_candidates': [h.to_dict() for h in admissible_hypotheses],
            'technical_indicators': {k: v.iloc[-1] if not v.empty else None 
                                  for k, v in self.current_technical_indicators.items()},
            'ml_predictions': [],  # Would be populated by ML predictor
            'best_hypothesis_id': best_hypothesis.id if best_hypothesis else None,
            'analysis_duration_ms': (
                datetime.datetime.now(datetime.timezone.utc) - analysis_start_time
            ).total_seconds() * 1000
        }
        
        # Store analysis result
        self.data_storage.store_wave_analysis_result(result)
        
        self.last_analysis_time = analysis_start_time
        logger.info(f"Market analysis completed in {result['analysis_duration_ms']:.2f} ms")
        
        return result
        
    def execute_trading_logic(self, current_price: float) -> List[str]:
        """
        Execute trading logic based on current analysis and market data.
        
        Args:
            current_price: Current market price
            
        Returns:
            List of action messages
        """
        actions = []
        
        # Check existing positions for stop loss/take profit hits
        position_actions = self.trade_executor.check_positions(current_price)
        actions.extend(position_actions)
        
        # Get the best hypothesis
        best_hypothesis = self.hypothesis_engine.get_best_hypothesis()
        
        if best_hypothesis:
            # Check if we should enter a new trade
            # This would involve confidence thresholds and other criteria
            confidence_threshold = self.config.get('trading', {}).get('min_confidence_threshold', 0.65)
            
            if (best_hypothesis.ml_ranking_score >= confidence_threshold and 
                best_hypothesis.confidence_score >= 0.8):  # High rule compliance
                
                # Check if we can execute a trade
                if self.trade_executor.can_execute_trade():
                    # Execute trade based on best hypothesis
                    position_id = self.trade_executor.execute_trade(
                        best_hypothesis, 
                        self.current_market_data, 
                        current_price
                    )
                    
                    if position_id:
                        actions.append(f"Executed trade {position_id} based on hypothesis {best_hypothesis.id}")
                        logger.info(f"Executed trade {position_id}")
                        
                        # Store the new position
                        if position_id in self.trade_executor.positions:
                            position_data = self.trade_executor.positions[position_id].to_dict()
                            self.data_storage.store_trade_position(position_data)
        
        # Store any closed positions
        # (Already done in check_positions and check_for_invalidation_exits)
        
        return actions
        
    def get_system_status(self) -> Dict:
        """
        Get current system status including hypotheses, positions, and account info.
        
        Returns:
            Dictionary with system status information
        """
        # Get active hypotheses
        active_hypotheses = self.hypothesis_engine.get_active_hypotheses()
        
        # Get open positions
        open_positions = self.trade_executor.get_open_positions()
        
        # Get account summary
        account_summary = self.trade_executor.get_account_summary()
        
        # Get database stats
        db_stats = self.data_storage.get_database_stats()
        
        return {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'system_running': self.is_running,
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'active_hypotheses': len(active_hypotheses),
            'open_positions': len(open_positions),
            'account_summary': account_summary,
            'database_stats': db_stats,
            'current_price': self.current_market_data.iloc[-1]['price'] if self.current_market_data is not None and not self.current_market_data.empty else None
        }
        
    def start_system(self):
        """Start the integrated trading system."""
        self.is_running = True
        logger.info("Integrated trading system started")
        
    def stop_system(self):
        """Stop the integrated trading system."""
        self.is_running = False
        logger.info("Integrated trading system stopped")
        
    def serialize_full_state(self) -> Dict:
        """
        Serialize the complete state of the trading system.
        
        Returns:
            Dictionary with complete system state
        """
        return {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'hypothesis_engine': self.hypothesis_engine.serialize_state(),
            'trade_executor': self.trade_executor.serialize_state(),
            'is_running': self.is_running,
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None
        }
        
    def restore_full_state(self, state: Dict):
        """
        Restore the complete state of the trading system.
        
        Args:
            state: Dictionary with complete system state
        """
        # Restore hypothesis engine
        if 'hypothesis_engine' in state:
            self.hypothesis_engine.restore_state(state['hypothesis_engine'])
            
        # Restore trade executor
        if 'trade_executor' in state:
            self.trade_executor.restore_state(state['trade_executor'])
            
        # Restore other state
        self.is_running = state.get('is_running', False)
        
        if state.get('last_analysis_time'):
            self.last_analysis_time = datetime.datetime.fromisoformat(
                state['last_analysis_time'].replace('Z', '+00:00')
            )
            
        logger.info("Full system state restored")

# Example usage function
def run_trading_session(config: Dict, price_data: pd.DataFrame):
    """
    Example function to run a trading session.
    
    Args:
        config: Configuration dictionary
        price_data: DataFrame with price data
    """
    # Initialize system
    system = IntegratedTradingSystem(config)
    system.start_system()
    
    try:
        # Perform initial analysis
        analysis_result = system.analyze_market_data(price_data)
        print(f"Analysis completed: {len(analysis_result['wave_candidates'])} candidates")
        
        # Execute trading logic (using last price)
        if not price_data.empty:
            last_price = price_data.iloc[-1]['price']
            actions = system.execute_trading_logic(last_price)
            for action in actions:
                print(f"Action: {action}")
        
        # Print status
        status = system.get_system_status()
        print(f"System status: {status['active_hypotheses']} hypotheses, {status['open_positions']} positions")
        
    finally:
        system.stop_system()
        
    return system