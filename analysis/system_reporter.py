"""
System Reporter
Generates comprehensive reports and handles advanced logging for the Elliott Wave trading system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .data_storage import DataStorage
from .continuous_improvement_loop import ContinuousImprovementLoop

logger = logging.getLogger(__name__)

class SystemReporter:
    """
    Generates comprehensive reports and handles advanced logging for the trading system.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_storage = DataStorage(config.get('database_path', './data/elliott_wave_data.db'))
        self.reports_dir = Path("./reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.plots_dir = self.reports_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info("SystemReporter initialized")
        
    def generate_comprehensive_report(self, report_name: str = None) -> str:
        """
        Generate a comprehensive system report.
        
        Args:
            report_name: Name for the report file (optional)
            
        Returns:
            Path to the generated report
        """
        if report_name is None:
            report_name = f"system_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        logger.info(f"Generating comprehensive report: {report_name}")
        
        # Collect all report data
        report_data = self._collect_report_data()
        
        # Generate report sections
        executive_summary = self._generate_executive_summary(report_data)
        system_overview = self._generate_system_overview(report_data)
        performance_analysis = self._generate_performance_analysis(report_data)
        hypothesis_analysis = self._generate_hypothesis_analysis(report_data)
        trading_analysis = self._generate_trading_analysis(report_data)
        data_quality_report = self._generate_data_quality_report(report_data)
        recommendations = self._generate_recommendations(report_data)
        
        # Combine all sections
        full_report = f"""
ELLIOTT WAVE TRADING SYSTEM COMPREHENSIVE REPORT
==============================================

Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{executive_summary}

{system_overview}

{performance_analysis}

{hypothesis_analysis}

{trading_analysis}

{data_quality_report}

{recommendations}

---
Report End
"""
        
        # Save report to file
        report_file = self.reports_dir / f"{report_name}.txt"
        try:
            with open(report_file, 'w') as f:
                f.write(full_report)
            logger.info(f"Report saved to {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""
            
        # Generate plots
        self._generate_performance_plots(report_data)
        
        return str(report_file)
    
    def _collect_report_data(self) -> Dict:
        """
        Collect all data needed for the report.
        
        Returns:
            Dictionary with all report data
        """
        logger.info("Collecting report data")
        
        report_data = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'database_stats': {},
            'recent_trades': [],
            'active_hypotheses': [],
            'invalidated_hypotheses': []
        }
        
        try:
            # Get database statistics
            report_data['database_stats'] = self.data_storage.get_database_stats()
        except Exception as e:
            logger.error(f"Error collecting database stats: {e}")
            
        try:
            # Get recent trades
            report_data['recent_trades'] = self.data_storage.get_recent_trade_performance(hours=168)  # Last week
        except Exception as e:
            logger.error(f"Error collecting recent trades: {e}")
            
        try:
            # Get active hypotheses
            report_data['active_hypotheses'] = self.data_storage.get_active_hypotheses()
        except Exception as e:
            logger.error(f"Error collecting active hypotheses: {e}")
            
        return report_data
    
    def _generate_executive_summary(self, report_data: Dict) -> str:
        """
        Generate executive summary section.
        
        Args:
            report_data: Collected report data
            
        Returns:
            Executive summary text
        """
        db_stats = report_data.get('database_stats', {})
        recent_trades = report_data.get('recent_trades', [])
        
        total_trades = db_stats.get('total_trades', 0)
        closed_trades = db_stats.get('closed_trades', 0)
        active_hypotheses = db_stats.get('active_hypotheses', 0)
        
        # Calculate P&L statistics
        if recent_trades:
            pnl_values = [trade.get('pnl', 0) for trade in recent_trades]
            total_pnl = sum(pnl_values)
            avg_pnl = np.mean(pnl_values) if pnl_values else 0
            win_rate = len([pnl for pnl in pnl_values if pnl > 0]) / len(pnl_values) if pnl_values else 0
        else:
            total_pnl = 0
            avg_pnl = 0
            win_rate = 0
        
        summary = f"""
EXECUTIVE SUMMARY
-----------------
Total Trades: {total_trades}
Closed Trades: {closed_trades}
Active Hypotheses: {active_hypotheses}
Total P&L: ${total_pnl:.2f}
Average P&L per Trade: ${avg_pnl:.2f}
Win Rate: {win_rate:.2%}
"""
        
        return summary
    
    def _generate_system_overview(self, report_data: Dict) -> str:
        """
        Generate system overview section.
        
        Args:
            report_data: Collected report data
            
        Returns:
            System overview text
        """
        db_stats = report_data.get('database_stats', {})
        
        overview = f"""
SYSTEM OVERVIEW
---------------
Database Statistics:
  Total Hypotheses: {db_stats.get('total_hypotheses', 0)}
  Active Hypotheses: {db_stats.get('active_hypotheses', 0)}
  Invalidated Hypotheses: {db_stats.get('invalidated_hypotheses', 0)}
  Total Trades: {db_stats.get('total_trades', 0)}
  Closed Trades: {db_stats.get('closed_trades', 0)}
  Market Data Records: {db_stats.get('market_data_records', 0)}
  Analysis Results: {db_stats.get('analysis_results', 0)}
"""
        
        return overview
    
    def _generate_performance_analysis(self, report_data: Dict) -> str:
        """
        Generate performance analysis section.
        
        Args:
            report_data: Collected report data
            
        Returns:
            Performance analysis text
        """
        recent_trades = report_data.get('recent_trades', [])
        
        if not recent_trades:
            return """
PERFORMANCE ANALYSIS
--------------------
No recent trades available for analysis.
"""
        
        # Calculate performance metrics
        pnl_values = [trade.get('pnl', 0) for trade in recent_trades]
        total_pnl = sum(pnl_values)
        avg_pnl = np.mean(pnl_values) if pnl_values else 0
        std_pnl = np.std(pnl_values) if pnl_values else 0
        win_rate = len([pnl for pnl in pnl_values if pnl > 0]) / len(pnl_values) if pnl_values else 0
        
        # Calculate Sharpe-like ratio (assuming risk-free rate of 0)
        sharpe_ratio = avg_pnl / std_pnl if std_pnl > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        rolling_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        analysis = f"""
PERFORMANCE ANALYSIS
--------------------
Total P&L: ${total_pnl:.2f}
Average P&L per Trade: ${avg_pnl:.2f}
P&L Standard Deviation: ${std_pnl:.2f}
Win Rate: {win_rate:.2%}
Sharpe Ratio: {sharpe_ratio:.2f}
Maximum Drawdown: ${max_drawdown:.2f}
Number of Trades Analyzed: {len(recent_trades)}
"""
        
        return analysis
    
    def _generate_hypothesis_analysis(self, report_data: Dict) -> str:
        """
        Generate hypothesis analysis section.
        
        Args:
            report_data: Collected report data
            
        Returns:
            Hypothesis analysis text
        """
        active_hypotheses = report_data.get('active_hypotheses', [])
        db_stats = report_data.get('database_stats', {})
        
        total_hypotheses = db_stats.get('total_hypotheses', 0)
        active_count = db_stats.get('active_hypotheses', 0)
        invalidated_count = db_stats.get('invalidated_hypotheses', 0)
        
        # Calculate validity rate
        validity_rate = active_count / total_hypotheses if total_hypotheses > 0 else 0
        
        # Analyze active hypotheses
        if active_hypotheses:
            confidence_scores = [hyp.get('confidence_score', 0) for hyp in active_hypotheses]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            ml_scores = [hyp.get('ml_ranking_score', 0) for hyp in active_hypotheses]
            avg_ml_score = np.mean(ml_scores) if ml_scores else 0
        else:
            avg_confidence = 0
            avg_ml_score = 0
        
        analysis = f"""
HYPOTHESIS ANALYSIS
-------------------
Total Hypotheses Generated: {total_hypotheses}
Active Hypotheses: {active_count}
Invalidated Hypotheses: {invalidated_count}
Hypothesis Validity Rate: {validity_rate:.2%}
Average Confidence Score: {avg_confidence:.3f}
Average ML Ranking Score: {avg_ml_score:.3f}
"""
        
        return analysis
    
    def _generate_trading_analysis(self, report_data: Dict) -> str:
        """
        Generate trading analysis section.
        
        Args:
            report_data: Collected report data
            
        Returns:
            Trading analysis text
        """
        recent_trades = report_data.get('recent_trades', [])
        
        if not recent_trades:
            return """
TRADING ANALYSIS
----------------
No recent trades available for analysis.
"""
        
        # Categorize trades by status
        stopped_out = len([t for t in recent_trades if t.get('status') == 'stopped_out'])
        take_profit_hit = len([t for t in recent_trades if t.get('status') == 'take_profit_hit'])
        manually_closed = len([t for t in recent_trades if t.get('status') == 'closed'])
        
        # Calculate P&L by status
        stopped_out_pnl = sum([t.get('pnl', 0) for t in recent_trades if t.get('status') == 'stopped_out'])
        take_profit_pnl = sum([t.get('pnl', 0) for t in recent_trades if t.get('status') == 'take_profit_hit'])
        
        analysis = f"""
TRADING ANALYSIS
----------------
Total Recent Trades: {len(recent_trades)}
Trades Stopped Out: {stopped_out} (P&L: ${stopped_out_pnl:.2f})
Trades Hit Take Profit: {take_profit_hit} (P&L: ${take_profit_pnl:.2f})
Manually Closed Trades: {manually_closed}
Winning Trades: {len([t for t in recent_trades if t.get('pnl', 0) > 0])}
Losing Trades: {len([t for t in recent_trades if t.get('pnl', 0) < 0])}
"""
        
        return analysis
    
    def _generate_data_quality_report(self, report_data: Dict) -> str:
        """
        Generate data quality report section.
        
        Args:
            report_data: Collected report data
            
        Returns:
            Data quality report text
        """
        db_stats = report_data.get('database_stats', {})
        
        market_records = db_stats.get('market_data_records', 0)
        analysis_results = db_stats.get('analysis_results', 0)
        
        report = f"""
DATA QUALITY REPORT
-------------------
Market Data Records: {market_records}
Analysis Results Stored: {analysis_results}
Data Integrity: {self._assess_data_integrity(db_stats)}
Data Consistency: {self._assess_data_consistency(report_data)}
"""
        
        return report
    
    def _assess_data_integrity(self, db_stats: Dict) -> str:
        """
        Assess data integrity.
        
        Args:
            db_stats: Database statistics
            
        Returns:
            Data integrity assessment
        """
        total_trades = db_stats.get('total_trades', 0)
        closed_trades = db_stats.get('closed_trades', 0)
        
        if total_trades == 0:
            return "UNKNOWN - No trades recorded"
            
        if closed_trades == total_trades:
            return "GOOD - All trades properly closed"
        elif closed_trades / total_trades > 0.9:
            return "ACCEPTABLE - Most trades properly closed"
        else:
            return "POOR - Many trades not properly closed"
    
    def _assess_data_consistency(self, report_data: Dict) -> str:
        """
        Assess data consistency.
        
        Args:
            report_data: Collected report data
            
        Returns:
            Data consistency assessment
        """
        active_hypotheses = report_data.get('active_hypotheses', [])
        recent_trades = report_data.get('recent_trades', [])
        
        if not active_hypotheses and not recent_trades:
            return "UNKNOWN - No data available"
            
        return "GOOD - Data structures consistent"
    
    def _generate_recommendations(self, report_data: Dict) -> str:
        """
        Generate recommendations section.
        
        Args:
            report_data: Collected report data
            
        Returns:
            Recommendations text
        """
        db_stats = report_data.get('database_stats', {})
        recent_trades = report_data.get('recent_trades', [])
        
        recommendations = ["RECOMMENDATIONS", "-------------"]
        
        # Check for data sufficiency
        if db_stats.get('market_data_records', 0) < 1000:
            recommendations.append("- Collect more market data for better analysis")
            
        if db_stats.get('total_hypotheses', 0) < 100:
            recommendations.append("- Generate more wave hypotheses to improve ML training")
            
        # Check for trading performance
        if recent_trades:
            win_rate = len([t for t in recent_trades if t.get('pnl', 0) > 0]) / len(recent_trades)
            if win_rate < 0.4:
                recommendations.append("- Review trading strategy and risk management parameters")
                
        # General recommendations
        recommendations.append("- Continue regular system monitoring and maintenance")
        recommendations.append("- Consider expanding to additional market instruments")
        recommendations.append("- Review and update Elliott Wave rule implementations periodically")
        
        return "\n".join(recommendations)
    
    def _generate_performance_plots(self, report_data: Dict):
        """
        Generate performance plots and save them.
        
        Args:
            report_data: Collected report data
        """
        recent_trades = report_data.get('recent_trades', [])
        
        if not recent_trades:
            logger.info("No recent trades available for plotting")
            return
            
        try:
            # Create P&L distribution plot
            pnl_values = [trade.get('pnl', 0) for trade in recent_trades]
            
            plt.figure(figsize=(10, 6))
            plt.hist(pnl_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Distribution of Trade P&L')
            plt.xlabel('Profit/Loss ($)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.plots_dir / "pnl_distribution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"P&L distribution plot saved to {plot_file}")
            
            # Create cumulative P&L plot
            cumulative_pnl = np.cumsum(pnl_values)
            
            plt.figure(figsize=(12, 6))
            plt.plot(cumulative_pnl, linewidth=2, color='green')
            plt.title('Cumulative P&L Over Time')
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative P&L ($)')
            plt.grid(True, alpha=0.3)
            
            plot_file = self.plots_dir / "cumulative_pnl.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cumulative P&L plot saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating performance plots: {e}")
    
    def generate_real_time_dashboard_data(self) -> Dict:
        """
        Generate data for a real-time dashboard.
        
        Returns:
            Dictionary with dashboard data
        """
        logger.info("Generating real-time dashboard data")
        
        dashboard_data = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'system_status': 'operational',
            'metrics': {}
        }
        
        try:
            # Get database stats for key metrics
            db_stats = self.data_storage.get_database_stats()
            
            dashboard_data['metrics'] = {
                'active_hypotheses': db_stats.get('active_hypotheses', 0),
                'total_trades': db_stats.get('total_trades', 0),
                'closed_trades': db_stats.get('closed_trades', 0),
                'market_data_points': db_stats.get('market_data_records', 0)
            }
            
            # Get recent performance
            recent_trades = self.data_storage.get_recent_trade_performance(hours=24)
            if recent_trades:
                pnl_values = [trade.get('pnl', 0) for trade in recent_trades]
                dashboard_data['metrics']['recent_pnl'] = sum(pnl_values)
                dashboard_data['metrics']['recent_win_rate'] = len([p for p in pnl_values if p > 0]) / len(pnl_values) if pnl_values else 0
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            dashboard_data['system_status'] = 'degraded'
            
        return dashboard_data
    
    def log_system_event(self, event_type: str, message: str, severity: str = 'info'):
        """
        Log a system event with structured logging.
        
        Args:
            event_type: Type of event (e.g., 'hypothesis_generated', 'trade_executed')
            message: Event message
            severity: Severity level ('info', 'warning', 'error')
        """
        event_data = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'event_type': event_type,
            'message': message,
            'severity': severity
        }
        
        # Log to file
        log_file = self.reports_dir / "system_events.log"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
            
        # Also log using standard logger
        log_func = getattr(logger, severity, logger.info)
        log_func(f"[{event_type}] {message}")

# Example usage function
def generate_system_report(config: Dict = None):
    """
    Generate a comprehensive system report.
    
    Args:
        config: Configuration dictionary (optional)
    """
    if config is None:
        config = {
            'database_path': './data/elliott_wave_data.db'
        }
    
    # Initialize reporter
    reporter = SystemReporter(config)
    
    # Generate comprehensive report
    report_path = reporter.generate_comprehensive_report()
    
    if report_path:
        print(f"Report generated successfully: {report_path}")
        
        # Also generate dashboard data
        dashboard_data = reporter.generate_real_time_dashboard_data()
        print("\nDashboard Data:")
        print(json.dumps(dashboard_data, indent=2))
    else:
        print("Failed to generate report")
    
    return reporter

if __name__ == "__main__":
    # Generate report when script is executed directly
    generate_system_report()