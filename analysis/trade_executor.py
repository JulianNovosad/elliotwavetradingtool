"""
Trade Executor
Executes trades based on valid wave hypotheses with proper risk management.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
import json

from .wave_hypothesis_engine import WaveHypothesis

logger = logging.getLogger(__name__)

class Position:
    """
    Represents a trading position.
    """
    def __init__(self, symbol: str, direction: str, quantity: float, entry_price: float, 
                 stop_loss: float, take_profit: float, hypothesis_id: str):
        self.symbol = symbol
        self.direction = direction  # 'long' or 'short'
        self.quantity = quantity
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.hypothesis_id = hypothesis_id
        self.entry_time = datetime.datetime.now(datetime.timezone.utc)
        self.status = 'open'  # 'open', 'closed', 'stopped_out', 'take_profit_hit'
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        
    def update_with_price(self, current_price: float) -> Optional[str]:
        """
        Update position with current price and check for stop loss or take profit.
        
        Args:
            current_price: Current market price
            
        Returns:
            Status change message or None if no change
        """
        if self.status != 'open':
            return None
            
        if self.direction == 'long':
            # Check stop loss (price went below stop loss)
            if current_price <= self.stop_loss:
                self.status = 'stopped_out'
                self.exit_price = self.stop_loss
                self.exit_time = datetime.datetime.now(datetime.timezone.utc)
                self.pnl = (self.stop_loss - self.entry_price) * self.quantity
                return f"Position stopped out at {self.stop_loss}"
                
            # Check take profit (price went above take profit)
            elif current_price >= self.take_profit:
                self.status = 'take_profit_hit'
                self.exit_price = self.take_profit
                self.exit_time = datetime.datetime.now(datetime.timezone.utc)
                self.pnl = (self.take_profit - self.entry_price) * self.quantity
                return f"Position hit take profit at {self.take_profit}"
                
        elif self.direction == 'short':
            # Check stop loss (price went above stop loss)
            if current_price >= self.stop_loss:
                self.status = 'stopped_out'
                self.exit_price = self.stop_loss
                self.exit_time = datetime.datetime.now(datetime.timezone.utc)
                self.pnl = (self.entry_price - self.stop_loss) * self.quantity
                return f"Position stopped out at {self.stop_loss}"
                
            # Check take profit (price went below take profit)
            elif current_price <= self.take_profit:
                self.status = 'take_profit_hit'
                self.exit_price = self.take_profit
                self.exit_time = datetime.datetime.now(datetime.timezone.utc)
                self.pnl = (self.entry_price - self.take_profit) * self.quantity
                return f"Position hit take profit at {self.take_profit}"
                
        return None
        
    def close_position(self, current_price: float) -> str:
        """
        Close position at current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            Status message
        """
        if self.status != 'open':
            return f"Position already {self.status}"
            
        self.status = 'closed'
        self.exit_price = current_price
        self.exit_time = datetime.datetime.now(datetime.timezone.utc)
        
        if self.direction == 'long':
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:  # short
            self.pnl = (self.entry_price - current_price) * self.quantity
            
        return f"Position closed at {current_price}"
        
    def to_dict(self) -> Dict:
        """Convert position to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'hypothesis_id': self.hypothesis_id,
            'entry_time': self.entry_time.isoformat(),
            'status': self.status,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl
        }

class TradeExecutor:
    """
    Executes trades based on valid wave hypotheses with risk management.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.positions = {}  # position_id -> Position
        self.trade_history = []  # List of closed positions
        self.account_balance = config.get('initial_balance', 10000.0)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% of account
        self.max_positions = config.get('max_positions', 5)
        self.current_risk = 0.0
        
        logger.info("TradeExecutor initialized")
        
    def can_execute_trade(self) -> bool:
        """
        Check if we can execute a new trade based on risk and position limits.
        
        Returns:
            True if we can execute a trade, False otherwise
        """
        # Check position limit
        open_positions = sum(1 for p in self.positions.values() if p.status == 'open')
        if open_positions >= self.max_positions:
            logger.debug(f"Position limit reached: {open_positions}/{self.max_positions}")
            return False
            
        # Check risk limit
        if self.current_risk >= self.account_balance * self.max_risk_per_trade * self.max_positions:
            logger.debug(f"Risk limit reached: {self.current_risk}")
            return False
            
        return True
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            symbol: Trading symbol
            
        Returns:
            Position size (quantity)
        """
        # Calculate risk per share/contract
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            logger.warning("Stop loss must be different from entry price")
            return 0.0
            
        # Calculate maximum risk for this trade
        max_trade_risk = self.account_balance * self.max_risk_per_trade
        
        # Calculate position size
        position_size = max_trade_risk / risk_per_share
        
        # Apply any symbol-specific limits
        max_position_size = self.config.get('symbol_limits', {}).get(symbol, {}).get('max_position_size')
        if max_position_size and position_size > max_position_size:
            position_size = max_position_size
            
        logger.debug(f"Calculated position size: {position_size} for {symbol} with risk {max_trade_risk}")
        return position_size
        
    def execute_trade(self, hypothesis: WaveHypothesis, market_data: pd.DataFrame, 
                     current_price: float) -> Optional[str]:
        """
        Execute a trade based on a valid wave hypothesis.
        
        Args:
            hypothesis: Valid WaveHypothesis to base trade on
            market_data: Current market data
            current_price: Current market price
            
        Returns:
            Position ID if trade executed, None if not
        """
        # Check if we can execute a trade
        if not self.can_execute_trade():
            logger.debug("Cannot execute trade - limits reached")
            return None
            
        # Determine trade direction based on wave pattern
        direction = self._determine_trade_direction(hypothesis, current_price)
        if not direction:
            logger.debug("Could not determine trade direction from hypothesis")
            return None
            
        # Calculate stop loss and take profit levels
        stop_loss, take_profit = self._calculate_risk_levels(hypothesis, current_price, direction)
        if not stop_loss or not take_profit:
            logger.warning("Could not calculate risk levels")
            return None
            
        # Calculate position size
        position_size = self.calculate_position_size(current_price, stop_loss, "SYMBOL")
        if position_size <= 0:
            logger.warning("Invalid position size calculated")
            return None
            
        # Create position
        position_id = f"pos_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.positions)}"
        position = Position(
            symbol="SYMBOL",  # Would be dynamic in real implementation
            direction=direction,
            quantity=position_size,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            hypothesis_id=hypothesis.id
        )
        
        # Update current risk
        self.current_risk += abs(current_price - stop_loss) * position_size
        
        # Store position
        self.positions[position_id] = position
        
        logger.info(f"Executed {direction} position {position_id} for hypothesis {hypothesis.id}")
        return position_id
        
    def _determine_trade_direction(self, hypothesis: WaveHypothesis, current_price: float) -> Optional[str]:
        """
        Determine trade direction from wave hypothesis.
        
        Args:
            hypothesis: WaveHypothesis to analyze
            current_price: Current market price
            
        Returns:
            'long', 'short', or None if direction cannot be determined
        """
        wave_pattern = hypothesis.wave_count.get('wave_pattern', {})
        label = wave_pattern.get('label', '')
        
        # Simple logic based on wave pattern labels
        if label.startswith('1-2-3-4-5'):
            # In an impulse wave, direction depends on which wave we're in
            # This is a simplified approach - real implementation would be more complex
            return 'long'  # Assume continuation of uptrend
        elif label.startswith('a-b-c'):
            # In a corrective wave, direction depends on context
            # This is a simplified approach - real implementation would be more complex
            return 'short'  # Assume this is a downward correction
        else:
            # Could not determine direction
            return None
            
    def _calculate_risk_levels(self, hypothesis: WaveHypothesis, current_price: float, 
                              direction: str) -> tuple:
        """
        Calculate stop loss and take profit levels based on wave analysis.
        
        Args:
            hypothesis: WaveHypothesis to analyze
            current_price: Current market price
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Tuple of (stop_loss, take_profit) prices
        """
        # Simple fixed percentage approach - real implementation would use wave structure
        stop_loss_pct = 0.02  # 2% stop loss
        take_profit_pct = 0.04  # 4% take profit
        
        if direction == 'long':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # short
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
            
        return (stop_loss, take_profit)
        
    def check_positions(self, current_price: float) -> List[str]:
        """
        Check all open positions for stop loss or take profit hits.
        
        Args:
            current_price: Current market price
            
        Returns:
            List of status change messages
        """
        status_changes = []
        
        # Create a list of keys to avoid modifying dict during iteration
        position_ids = list(self.positions.keys())
        
        for position_id in position_ids:
            position = self.positions[position_id]
            if position.status == 'open':
                status_msg = position.update_with_price(current_price)
                if status_msg:
                    status_changes.append(status_msg)
                    # Update account balance
                    self.account_balance += position.pnl
                    # Update current risk
                    risk_amount = abs(position.entry_price - position.stop_loss) * position.quantity
                    self.current_risk -= risk_amount
                    # Move to trade history
                    self.trade_history.append(position.to_dict())
                    logger.info(f"Position {position_id} closed: {status_msg}")
                    
        return status_changes
        
    def check_for_invalidation_exits(self, invalidated_hypotheses: List[str]) -> List[str]:
        """
        Check if any open positions need to be exited due to hypothesis invalidation.
        
        Args:
            invalidated_hypotheses: List of invalidated hypothesis IDs
            
        Returns:
            List of exit messages
        """
        exit_messages = []
        
        # Create a list of keys to avoid modifying dict during iteration
        position_ids = list(self.positions.keys())
        
        for position_id in position_ids:
            position = self.positions[position_id]
            if position.status == 'open' and position.hypothesis_id in invalidated_hypotheses:
                # Exit position immediately
                exit_msg = position.close_position(position.entry_price)  # Use entry price for immediate exit
                exit_messages.append(f"Position {position_id} exited due to hypothesis invalidation: {exit_msg}")
                
                # Update account balance
                self.account_balance += position.pnl
                # Update current risk
                risk_amount = abs(position.entry_price - position.stop_loss) * position.quantity
                self.current_risk -= risk_amount
                # Move to trade history
                self.trade_history.append(position.to_dict())
                logger.info(f"Position {position_id} exited due to invalidation: {exit_msg}")
                
        return exit_messages
        
    def get_open_positions(self) -> List[Position]:
        """Get all currently open positions."""
        return [p for p in self.positions.values() if p.status == 'open']
        
    def get_account_summary(self) -> Dict:
        """Get account summary information."""
        open_positions = self.get_open_positions()
        closed_trades = len(self.trade_history)
        total_pnl = sum(pos.pnl for pos in self.trade_history)
        
        return {
            'account_balance': self.account_balance,
            'current_risk': self.current_risk,
            'open_positions': len(open_positions),
            'closed_trades': closed_trades,
            'total_pnl': total_pnl,
            'win_rate': self._calculate_win_rate()
        }
        
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self.trade_history:
            return 0.0
            
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return winning_trades / len(self.trade_history)
        
    def serialize_state(self) -> Dict:
        """Serialize the current state of the trade executor."""
        return {
            'positions': {pid: pos.to_dict() for pid, pos in self.positions.items()},
            'trade_history': self.trade_history,
            'account_balance': self.account_balance,
            'current_risk': self.current_risk,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
    def restore_state(self, state: Dict):
        """Restore the trade executor state from serialized data."""
        # Clear current state
        self.positions.clear()
        self.trade_history.clear()
        
        # Restore account state
        self.account_balance = state.get('account_balance', 10000.0)
        self.current_risk = state.get('current_risk', 0.0)
        
        # Restore positions
        for position_id, position_data in state.get('positions', {}).items():
            position = Position(
                symbol=position_data['symbol'],
                direction=position_data['direction'],
                quantity=position_data['quantity'],
                entry_price=position_data['entry_price'],
                stop_loss=position_data['stop_loss'],
                take_profit=position_data['take_profit'],
                hypothesis_id=position_data['hypothesis_id']
            )
            
            # Restore attributes
            position.status = position_data['status']
            position.exit_price = position_data['exit_price']
            position.pnl = position_data['pnl']
            
            # Parse timestamps
            position.entry_time = datetime.datetime.fromisoformat(
                position_data['entry_time'].replace('Z', '+00:00')
            )
            if position_data['exit_time']:
                position.exit_time = datetime.datetime.fromisoformat(
                    position_data['exit_time'].replace('Z', '+00:00')
                )
                
            self.positions[position_id] = position
            
        # Restore trade history
        self.trade_history = state.get('trade_history', [])
        
        logger.info(f"Restored trade executor state with {len(self.positions)} positions")