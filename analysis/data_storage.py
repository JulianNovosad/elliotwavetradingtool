"""
Data Storage
Handles storage of wave counts, hypothesis metadata, and trade outcomes.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
import json
import sqlite3
import os

logger = logging.getLogger(__name__)

class DataStorage:
    """
    Handles persistent storage of wave analysis data, hypotheses, and trade outcomes.
    """
    def __init__(self, db_path: str = "./data/elliott_wave_data.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"DataStorage initialized with database at {db_path}")
        
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create wave hypotheses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wave_hypotheses (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                is_valid BOOLEAN,
                wave_count JSON,
                segments JSON,
                rule_violations JSON,
                confidence_score REAL,
                ml_ranking_score REAL
            )
        ''')
        
        # Create invalidated hypotheses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS invalidated_hypotheses (
                id TEXT PRIMARY KEY,
                invalidated_at TIMESTAMP,
                wave_count JSON,
                rule_violations JSON,
                confidence_score REAL
            )
        ''')
        
        # Create trade positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_positions (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                quantity REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                hypothesis_id TEXT,
                entry_time TIMESTAMP,
                status TEXT,
                exit_price REAL,
                exit_time TIMESTAMP,
                pnl REAL
            )
        ''')
        
        # Create market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (timestamp, symbol)
            )
        ''')
        
        # Create wave analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wave_analysis_results (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                symbol TEXT,
                wave_candidates JSON,
                technical_indicators JSON,
                ml_predictions JSON,
                best_hypothesis_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def store_hypothesis(self, hypothesis_data: Dict):
        """
        Store a wave hypothesis in the database.
        
        Args:
            hypothesis_data: Dictionary containing hypothesis information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert JSON fields to strings
        wave_count_json = json.dumps(hypothesis_data.get('wave_count', {}))
        segments_json = json.dumps(hypothesis_data.get('segments', []))
        violations_json = json.dumps(hypothesis_data.get('rule_violations', []))
        
        cursor.execute('''
            INSERT OR REPLACE INTO wave_hypotheses 
            (id, created_at, last_updated, is_valid, wave_count, segments, rule_violations, 
             confidence_score, ml_ranking_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            hypothesis_data['id'],
            hypothesis_data['created_at'],
            hypothesis_data['last_updated'],
            hypothesis_data['is_valid'],
            wave_count_json,
            segments_json,
            violations_json,
            hypothesis_data.get('confidence_score', 0.0),
            hypothesis_data.get('ml_ranking_score', 0.0)
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Stored hypothesis {hypothesis_data['id']}")
        
    def store_invalidated_hypothesis(self, hypothesis_data: Dict):
        """
        Store an invalidated hypothesis in the database.
        
        Args:
            hypothesis_data: Dictionary containing invalidated hypothesis information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert JSON fields to strings
        wave_count_json = json.dumps(hypothesis_data.get('wave_count', {}))
        violations_json = json.dumps(hypothesis_data.get('rule_violations', []))
        
        cursor.execute('''
            INSERT OR REPLACE INTO invalidated_hypotheses 
            (id, invalidated_at, wave_count, rule_violations, confidence_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            hypothesis_data['id'],
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            wave_count_json,
            violations_json,
            hypothesis_data.get('confidence_score', 0.0)
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Stored invalidated hypothesis {hypothesis_data['id']}")
        
    def store_trade_position(self, position_data: Dict):
        """
        Store a trade position in the database.
        
        Args:
            position_data: Dictionary containing position information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trade_positions 
            (id, symbol, direction, quantity, entry_price, stop_loss, take_profit, 
             hypothesis_id, entry_time, status, exit_price, exit_time, pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position_data['id'],
            position_data['symbol'],
            position_data['direction'],
            position_data['quantity'],
            position_data['entry_price'],
            position_data['stop_loss'],
            position_data['take_profit'],
            position_data['hypothesis_id'],
            position_data['entry_time'],
            position_data['status'],
            position_data['exit_price'],
            position_data['exit_time'],
            position_data['pnl']
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Stored trade position {position_data['id']}")
        
    def store_market_data(self, df: pd.DataFrame, symbol: str):
        """
        Store market data in the database.
        
        Args:
            df: DataFrame with market data (timestamp index, OHLCV columns)
            symbol: Trading symbol
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for timestamp, row in df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (timestamp, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp.isoformat(),
                symbol,
                row.get('open', row.get('price', 0)),
                row.get('high', row.get('price', 0)),
                row.get('low', row.get('price', 0)),
                row.get('close', row.get('price', 0)),
                row.get('volume', 0)
            ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Stored {len(df)} market data records for {symbol}")
        
    def store_wave_analysis_result(self, analysis_data: Dict):
        """
        Store wave analysis result in the database.
        
        Args:
            analysis_data: Dictionary containing analysis results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert JSON fields to strings
        candidates_json = json.dumps(analysis_data.get('wave_candidates', []))
        indicators_json = json.dumps(analysis_data.get('technical_indicators', {}))
        predictions_json = json.dumps(analysis_data.get('ml_predictions', []))
        
        cursor.execute('''
            INSERT OR REPLACE INTO wave_analysis_results 
            (id, timestamp, symbol, wave_candidates, technical_indicators, ml_predictions, best_hypothesis_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_data['id'],
            analysis_data['timestamp'],
            analysis_data['symbol'],
            candidates_json,
            indicators_json,
            predictions_json,
            analysis_data.get('best_hypothesis_id')
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Stored wave analysis result {analysis_data['id']}")
        
    def get_active_hypotheses(self) -> List[Dict]:
        """
        Retrieve all active (valid) hypotheses from the database.
        
        Returns:
            List of hypothesis dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, created_at, last_updated, is_valid, wave_count, segments, 
                   rule_violations, confidence_score, ml_ranking_score
            FROM wave_hypotheses 
            WHERE is_valid = 1
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        hypotheses = []
        for row in rows:
            hypothesis = {
                'id': row[0],
                'created_at': row[1],
                'last_updated': row[2],
                'is_valid': row[3],
                'wave_count': json.loads(row[4]) if row[4] else {},
                'segments': json.loads(row[5]) if row[5] else [],
                'rule_violations': json.loads(row[6]) if row[6] else [],
                'confidence_score': row[7],
                'ml_ranking_score': row[8]
            }
            hypotheses.append(hypothesis)
            
        return hypotheses
        
    def get_recent_trade_performance(self, hours: int = 24) -> List[Dict]:
        """
        Retrieve recent trade performance data.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of trade dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
        
        cursor.execute('''
            SELECT id, symbol, direction, quantity, entry_price, stop_loss, take_profit, 
                   hypothesis_id, entry_time, status, exit_price, exit_time, pnl
            FROM trade_positions 
            WHERE entry_time >= ?
            ORDER BY entry_time DESC
        ''', (since_time.isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for row in rows:
            trade = {
                'id': row[0],
                'symbol': row[1],
                'direction': row[2],
                'quantity': row[3],
                'entry_price': row[4],
                'stop_loss': row[5],
                'take_profit': row[6],
                'hypothesis_id': row[7],
                'entry_time': row[8],
                'status': row[9],
                'exit_price': row[10],
                'exit_time': row[11],
                'pnl': row[12]
            }
            trades.append(trade)
            
        return trades
        
    def get_historical_analysis_results(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Retrieve historical wave analysis results.
        
        Args:
            symbol: Trading symbol
            days: Number of days to look back
            
        Returns:
            List of analysis result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
        
        cursor.execute('''
            SELECT id, timestamp, symbol, wave_candidates, technical_indicators, 
                   ml_predictions, best_hypothesis_id
            FROM wave_analysis_results 
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (symbol, since_time.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            analysis = {
                'id': row[0],
                'timestamp': row[1],
                'symbol': row[2],
                'wave_candidates': json.loads(row[3]) if row[3] else [],
                'technical_indicators': json.loads(row[4]) if row[4] else {},
                'ml_predictions': json.loads(row[5]) if row[5] else [],
                'best_hypothesis_id': row[6]
            }
            results.append(analysis)
            
        return results
        
    def cleanup_old_data(self, days: int = 30):
        """
        Clean up old data from the database.
        
        Args:
            days: Number of days to keep data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()
        
        # Delete old invalidated hypotheses
        cursor.execute('DELETE FROM invalidated_hypotheses WHERE invalidated_at < ?', (cutoff_str,))
        deleted_invalidated = cursor.rowcount
        
        # Delete old market data
        cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_str,))
        deleted_market = cursor.rowcount
        
        # Delete old analysis results
        cursor.execute('DELETE FROM wave_analysis_results WHERE timestamp < ?', (cutoff_str,))
        deleted_analysis = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up old data: {deleted_invalidated} invalidated hypotheses, "
                   f"{deleted_market} market records, {deleted_analysis} analysis results")
        
    def export_data_to_csv(self, table_name: str, csv_path: str):
        """
        Export data from a table to CSV format.
        
        Args:
            table_name: Name of the table to export
            csv_path: Path to save the CSV file
        """
        conn = sqlite3.connect(self.db_path)
        
        # Read table into DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        conn.close()
        
        logger.info(f"Exported {table_name} data to {csv_path}")
        
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database contents.
        
        Returns:
            Dictionary with database statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count hypotheses
        cursor.execute('SELECT COUNT(*) FROM wave_hypotheses')
        stats['total_hypotheses'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM wave_hypotheses WHERE is_valid = 1')
        stats['active_hypotheses'] = cursor.fetchone()[0]
        
        # Count invalidated hypotheses
        cursor.execute('SELECT COUNT(*) FROM invalidated_hypotheses')
        stats['invalidated_hypotheses'] = cursor.fetchone()[0]
        
        # Count trade positions
        cursor.execute('SELECT COUNT(*) FROM trade_positions')
        stats['total_trades'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM trade_positions WHERE status IN ("stopped_out", "take_profit_hit", "closed")')
        stats['closed_trades'] = cursor.fetchone()[0]
        
        # Count market data records
        cursor.execute('SELECT COUNT(*) FROM market_data')
        stats['market_data_records'] = cursor.fetchone()[0]
        
        # Count analysis results
        cursor.execute('SELECT COUNT(*) FROM wave_analysis_results')
        stats['analysis_results'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats