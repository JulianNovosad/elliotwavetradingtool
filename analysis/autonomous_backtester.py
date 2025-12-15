"""
Autonomous Backtester
Runs historical backtests on price data to validate the Elliott Wave trading system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
import json
import os
from pathlib import Path

from .integrated_trading_system import IntegratedTradingSystem
from .wave_hypothesis_engine import WaveHypothesis
from .data_storage import DataStorage

logger = logging.getLogger(__name__)

class AutonomousBacktester:
    """
    Runs autonomous backtests on historical price data to validate and improve the trading system.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_storage = DataStorage(config.get('database_path', './data/elliott_wave_data.db'))
        self.results = []
        self.performance_metrics = {}
        
        # Create results directory if it doesn't exist
        self.results_dir = Path("./backtest_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("AutonomousBacktester initialized")
        
    def load_historical_data(self, data_directory: str = "./data") -> Dict[str, pd.DataFrame]:
        """
        Load all historical CSV data files from the data directory.
        
        Args:
            data_directory: Path to directory containing CSV files
            
        Returns:
            Dictionary mapping timeframe names to DataFrames
        """
        data_files = Path(data_directory).glob("*.csv")
        historical_data = {}
        
        for file_path in data_files:
            try:
                # Extract timeframe from filename (e.g., sample_1h.csv -> 1h)
                timeframe = file_path.stem.split("_")[-1]
                
                # Load data
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Sort by timestamp
                df.sort_index(inplace=True)
                
                historical_data[timeframe] = df
                logger.info(f"Loaded {len(df)} records for timeframe {timeframe}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return historical_data
    
    def run_backtest(self, historical_data: Dict[str, pd.DataFrame], 
                     symbols: List[str] = None) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            historical_data: Dictionary of DataFrames with historical price data
            symbols: List of symbols to test (currently placeholder)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest")
        backtest_start_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Initialize results storage
        all_results = []
        
        # Test each timeframe
        for timeframe, df in historical_data.items():
            logger.info(f"Testing timeframe: {timeframe}")
            
            # Run sliding window analysis
            timeframe_results = self._run_sliding_window_analysis(df, timeframe)
            all_results.extend(timeframe_results)
            
        # Calculate overall performance metrics
        performance_metrics = self._calculate_performance_metrics(all_results)
        
        # Store results
        backtest_result = {
            'id': f"backtest_{backtest_start_time.strftime('%Y%m%d_%H%M%S')}",
            'start_time': backtest_start_time.isoformat(),
            'end_time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'timeframes_tested': list(historical_data.keys()),
            'total_tests': len(all_results),
            'results': all_results,
            'performance_metrics': performance_metrics
        }
        
        # Save results to file
        results_file = self.results_dir / f"backtest_{backtest_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(backtest_result, f, indent=2)
            
        logger.info(f"Backtest completed. Results saved to {results_file}")
        return backtest_result
    
    def _run_sliding_window_analysis(self, df: pd.DataFrame, timeframe: str, 
                                   window_size: int = 100) -> List[Dict]:
        """
        Run sliding window analysis on a DataFrame.
        
        Args:
            df: DataFrame with historical price data
            timeframe: Timeframe identifier
            window_size: Size of the sliding window
            
        Returns:
            List of analysis results
        """
        results = []
        
        # For smaller datasets, use a smaller window
        if len(df) < window_size:
            window_size = max(10, len(df) // 2)
            
        # Slide through the data
        for i in range(window_size, len(df), window_size // 4):  # 75% overlap
            # Get window data
            window_data = df.iloc[i-window_size:i]
            
            if len(window_data) < 10:  # Minimum data requirement
                continue
                
            # Run analysis on window
            try:
                window_result = self._analyze_window(window_data, timeframe, i)
                results.append(window_result)
            except Exception as e:
                logger.error(f"Error analyzing window at index {i}: {e}")
                
        return results
    
    def _analyze_window(self, window_data: pd.DataFrame, timeframe: str, 
                       window_index: int) -> Dict:
        """
        Analyze a single window of data.
        
        Args:
            window_data: DataFrame with window data
            timeframe: Timeframe identifier
            window_index: Index of the window
            
        Returns:
            Dictionary with analysis results
        """
        # Initialize trading system for this window
        system = IntegratedTradingSystem(self.config)
        
        # Run market analysis
        analysis_result = system.analyze_market_data(window_data)
        
        # Get current price (last price in window)
        current_price = window_data.iloc[-1]['price']
        
        # Execute trading logic
        actions = system.execute_trading_logic(current_price)
        
        # Get system status
        status = system.get_system_status()
        
        # Create window result
        window_result = {
            'window_index': window_index,
            'timeframe': timeframe,
            'timestamp': window_data.index[-1].isoformat(),
            'current_price': current_price,
            'active_hypotheses': status['active_hypotheses'],
            'open_positions': status['open_positions'],
            'account_balance': status['account_summary']['account_balance'],
            'total_pnl': status['account_summary']['total_pnl'],
            'actions_executed': len(actions),
            'analysis_duration_ms': analysis_result.get('analysis_duration_ms', 0)
        }
        
        return window_result
    
    def _calculate_performance_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results: List of backtest results
            
        Returns:
            Dictionary with performance metrics
        """
        if not results:
            return {}
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Calculate basic metrics
        total_pnl = df['total_pnl'].sum() if 'total_pnl' in df.columns else 0
        avg_pnl = df['total_pnl'].mean() if 'total_pnl' in df.columns else 0
        std_pnl = df['total_pnl'].std() if 'total_pnl' in df.columns else 0
        
        # Calculate Sharpe-like ratio (assuming risk-free rate of 0)
        sharpe_ratio = avg_pnl / std_pnl if std_pnl > 0 else 0
        
        # Calculate win rate
        winning_periods = len(df[df['total_pnl'] > 0]) if 'total_pnl' in df.columns else 0
        win_rate = winning_periods / len(df) if len(df) > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_pnl = df['total_pnl'].cumsum() if 'total_pnl' in df.columns else pd.Series([0])
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Calculate average analysis time
        avg_analysis_time = df['analysis_duration_ms'].mean() if 'analysis_duration_ms' in df.columns else 0
        
        metrics = {
            'total_pnl': total_pnl,
            'average_pnl': avg_pnl,
            'pnl_std_dev': std_pnl,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_periods': len(results),
            'winning_periods': winning_periods,
            'max_drawdown': max_drawdown,
            'avg_analysis_time_ms': avg_analysis_time,
            'timeframes_analyzed': df['timeframe'].unique().tolist() if 'timeframe' in df.columns else []
        }
        
        return metrics
    
    def run_continuous_improvement_loop(self, iterations: int = 5):
        """
        Run continuous improvement loop to refine the system.
        
        Args:
            iterations: Number of improvement iterations to run
        """
        logger.info(f"Starting continuous improvement loop for {iterations} iterations")
        
        for i in range(iterations):
            logger.info(f"Starting improvement iteration {i+1}/{iterations}")
            
            # Load historical data
            historical_data = self.load_historical_data()
            
            if not historical_data:
                logger.warning("No historical data found, skipping iteration")
                continue
                
            # Run backtest
            backtest_result = self.run_backtest(historical_data)
            
            # Store results
            self.results.append(backtest_result)
            
            # Analyze performance
            metrics = backtest_result.get('performance_metrics', {})
            logger.info(f"Iteration {i+1} results: PNL={metrics.get('total_pnl', 0):.2f}, "
                       f"Win Rate={metrics.get('win_rate', 0):.2%}")
            
            # In a real implementation, we would:
            # 1. Identify underperforming areas
            # 2. Adjust system parameters
            # 3. Retrain ML models
            # 4. Update rule engines
            # For now, we'll just log the intention
            
            logger.info(f"Completed improvement iteration {i+1}")
            
        logger.info("Continuous improvement loop completed")
        
    def generate_report(self) -> str:
        """
        Generate a summary report of all backtest results.
        
        Returns:
            String with summary report
        """
        if not self.results:
            return "No backtest results available"
            
        # Get latest result
        latest_result = self.results[-1]
        metrics = latest_result.get('performance_metrics', {})
        
        report = f"""
AUTONOMOUS BACKTEST REPORT
=========================

Test ID: {latest_result.get('id', 'N/A')}
Timeframes Tested: {', '.join(metrics.get('timeframes_analyzed', []))}
Total Periods: {metrics.get('total_periods', 0)}
Total PNL: ${metrics.get('total_pnl', 0):.2f}
Average PNL: ${metrics.get('average_pnl', 0):.2f}
Win Rate: {metrics.get('win_rate', 0):.2%}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: ${metrics.get('max_drawdown', 0):.2f}
Avg Analysis Time: {metrics.get('avg_analysis_time_ms', 0):.1f} ms

Database Statistics:
"""
        
        # Add database stats
        try:
            db_stats = self.data_storage.get_database_stats()
            for key, value in db_stats.items():
                report += f"  {key}: {value}\n"
        except Exception as e:
            report += f"  Error retrieving database stats: {e}\n"
            
        return report.strip()

# Example usage function
def run_autonomous_backtest(config: Dict = None):
    """
    Run an autonomous backtest with default configuration.
    
    Args:
        config: Configuration dictionary (optional)
    """
    if config is None:
        config = {
            'initial_balance': 10000.0,
            'max_risk_per_trade': 0.02,
            'max_positions': 5,
            'trading': {
                'min_confidence_threshold': 0.65
            }
        }
    
    # Initialize backtester
    backtester = AutonomousBacktester(config)
    
    # Load historical data
    historical_data = backtester.load_historical_data()
    
    if not historical_data:
        print("No historical data found!")
        return
        
    # Run backtest
    result = backtester.run_backtest(historical_data)
    
    # Generate report
    report = backtester.generate_report()
    print(report)
    
    # Run continuous improvement loop
    backtester.run_continuous_improvement_loop(iterations=3)
    
    return backtester

if __name__ == "__main__":
    # Run the backtest when script is executed directly
    run_autonomous_backtest()