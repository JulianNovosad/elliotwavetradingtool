"""
Main Autonomous Testing Script
Orchestrates the complete autonomous testing, validation, and improvement process.
"""

import logging
import datetime
import json
from pathlib import Path

from analysis.autonomous_backtester import AutonomousBacktester
from analysis.continuous_improvement_loop import ContinuousImprovementLoop
from analysis.system_reporter import SystemReporter
from analysis.integrated_trading_system import IntegratedTradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./autonomous_testing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the complete autonomous testing and improvement process.
    """
    logger.info("Starting autonomous Elliott Wave trading system testing and improvement")
    
    # Configuration
    config = {
        'database_path': './data/elliott_wave_data.db',
        'initial_balance': 10000.0,
        'max_risk_per_trade': 0.02,
        'max_positions': 5,
        'trading': {
            'min_confidence_threshold': 0.65
        }
    }
    
    try:
        # Phase 1: Backtesting
        logger.info("Phase 1: Running backtesting")
        backtester = AutonomousBacktester(config)
        historical_data = backtester.load_historical_data()
        
        if not historical_data:
            logger.error("No historical data found for backtesting")
            return
            
        backtest_result = backtester.run_backtest(historical_data)
        logger.info(f"Backtesting completed. Performance metrics: {backtest_result.get('performance_metrics', {})}")
        
        # Phase 2: Continuous Improvement
        logger.info("Phase 2: Running continuous improvement loop")
        improvement_loop = ContinuousImprovementLoop(config)
        improvement_loop.run_continuous_loop(cycles=2, interval_hours=1)
        
        # Phase 3: Reporting
        logger.info("Phase 3: Generating system report")
        reporter = SystemReporter(config)
        report_path = reporter.generate_comprehensive_report()
        
        if report_path:
            logger.info(f"Comprehensive report generated: {report_path}")
        else:
            logger.error("Failed to generate comprehensive report")
            
        # Generate dashboard data
        dashboard_data = reporter.generate_real_time_dashboard_data()
        logger.info(f"Dashboard data: {json.dumps(dashboard_data, indent=2)}")
        
        # Log completion
        logger.info("Autonomous testing and improvement process completed successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("AUTONOMOUS TESTING SUMMARY")
        print("="*60)
        print(f"Backtest ID: {backtest_result.get('id', 'N/A')}")
        metrics = backtest_result.get('performance_metrics', {})
        print(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Report Generated: {report_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in autonomous testing process: {e}")
        raise

if __name__ == "__main__":
    main()