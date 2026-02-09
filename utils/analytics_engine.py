import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from pathlib import Path
import json

DB_PATH = "trading_analytics.db"

def init_analytics_db():
    """Initialize SQLite database for tracking predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table 1: Store predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date_predicted TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verdict TEXT,
            predicted_price REAL,
            sl REAL,
            tgt REAL,
            confidence REAL,
            entry_price REAL,
            status TEXT DEFAULT 'OPEN'
        )
    ''')
    
    # Table 2: Track actual outcomes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            ticker TEXT,
            date_closed TIMESTAMP,
            exit_price REAL,
            pnl REAL,
            pnl_percent REAL,
            outcome TEXT,
            days_held INTEGER,
            hit_target BOOLEAN,
            hit_sl BOOLEAN,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )
    ''')
    
    # Table 3: Daily performance metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            total_signals INTEGER,
            buy_signals INTEGER,
            sell_signals INTEGER,
            hit_count INTEGER,
            miss_count INTEGER,
            win_rate REAL,
            avg_profit REAL,
            sharpe_ratio REAL
        )
    ''')
    
    conn.commit()
    conn.close()

@st.cache_resource
def get_db_connection():
    """Get cached database connection"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def log_prediction(ticker, verdict, predicted_price, sl, tgt, confidence, entry_price):
    """Log a new prediction"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions 
        (ticker, verdict, predicted_price, sl, tgt, confidence, entry_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, verdict, predicted_price, sl, tgt, confidence, entry_price))
    
    conn.commit()
    prediction_id = cursor.lastrowid
    
    return prediction_id

def close_prediction(prediction_id, exit_price, outcome):
    """Close a prediction with actual outcome"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get prediction details
    cursor.execute('''
        SELECT predicted_price, sl, tgt, date_predicted, entry_price 
        FROM predictions 
        WHERE id = ?
    ''', (prediction_id,))
    
    row = cursor.fetchone()
    if not row:
        return None
    
    predicted_price, sl, tgt, date_predicted, entry_price = row
    
    # Calculate metrics
    entry = entry_price if entry_price else predicted_price
    pnl = exit_price - entry
    pnl_percent = (pnl / entry) * 100 if entry > 0 else 0
    
    # Calculate days held
    pred_date = datetime.fromisoformat(date_predicted)
    days_held = (datetime.now() - pred_date).days
    
    # Determine if target/SL hit
    hit_target = exit_price >= tgt if pnl > 0 else exit_price <= tgt
    hit_sl = exit_price <= sl if pnl < 0 else False
    
    # Insert outcome
    cursor.execute('''
        INSERT INTO outcomes 
        (prediction_id, ticker, exit_price, pnl, pnl_percent, outcome, days_held, hit_target, hit_sl)
        SELECT id, ticker, ?, ?, ?, ?, ?, ?, ? 
        FROM predictions 
        WHERE id = ?
    ''', (exit_price, pnl, pnl_percent, outcome, days_held, hit_target, hit_sl, prediction_id))
    
    # Update prediction status
    cursor.execute('UPDATE predictions SET status = ? WHERE id = ?', ('CLOSED', prediction_id))
    
    conn.commit()
    
    return {
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'days_held': days_held,
        'hit_target': hit_target,
        'hit_sl': hit_sl
    }

def calculate_win_rate(days=30, ticker=None):
    """Calculate win rate for last N days"""
    conn = get_db_connection()
    
    query = '''
        SELECT 
            COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
            COUNT(CASE WHEN pnl < 0 THEN 1 END) as losses,
            COUNT(*) as total,
            AVG(pnl_percent) as avg_return,
            MAX(pnl_percent) as max_return,
            MIN(pnl_percent) as min_return
        FROM outcomes
        WHERE date_closed >= datetime('now', '-' || ? || ' days')
    '''
    
    params = [days]
    
    if ticker:
        query += ' AND ticker = ?'
        params.append(ticker)
    
    df = pd.read_sql_query(query, conn, params=params)
    
    conn.close()
    
    if df.empty or df['total'][0] == 0:
        return {
            'win_rate': 0,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'avg_return': 0,
            'profit_factor': 0
        }
    
    wins = df['wins'][0] or 0
    losses = df['losses'][0] or 0
    total = df['total'][0] or 1
    
    win_rate = (wins / total) * 100 if total > 0 else 0
    avg_return = df['avg_return'][0] or 0
    
    return {
        'win_rate': win_rate,
        'total_trades': int(total),
        'wins': int(wins),
        'losses': int(losses),
        'avg_return': avg_return,
        'max_return': df['max_return'][0] or 0,
        'min_return': df['min_return'][0] or 0
    }

def get_performance_metrics(days=30, ticker=None):
    """Get comprehensive performance metrics"""
    conn = get_db_connection()
    
    query = '''
        SELECT 
            pnl_percent,
            pnl,
            days_held,
            hit_target,
            hit_sl
        FROM outcomes
        WHERE date_closed >= datetime('now', '-' || ? || ' days')
    '''
    
    params = [days]
    if ticker:
        query += ' AND ticker = ?'
        params.append(ticker)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'avg_trade_duration': 0
        }
    
    # Calculate metrics
    total_trades = len(df)
    wins = len(df[df['pnl_percent'] > 0])
    losses = len(df[df['pnl_percent'] < 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Sharpe Ratio (assuming 252 trading days)
    returns = df['pnl_percent'].values
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    # Profit Factor (gross profit / gross loss)
    gross_profit = df[df['pnl_percent'] > 0]['pnl_percent'].sum()
    gross_loss = abs(df[df['pnl_percent'] < 0]['pnl_percent'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Max Drawdown (cumulative returns)
    cumulative = (1 + returns / 100).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Average trade duration
    avg_duration = df['days_held'].mean()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_trade_duration': avg_duration,
        'wins': wins,
        'losses': losses,
        'avg_win': df[df['pnl_percent'] > 0]['pnl_percent'].mean() if wins > 0 else 0,
        'avg_loss': df[df['pnl_percent'] < 0]['pnl_percent'].mean() if losses > 0 else 0,
    }

def get_daily_stats(date=None):
    """Get stats for a specific date"""
    if date is None:
        date = datetime.now().date()
    
    conn = get_db_connection()
    
    query = '''
        SELECT 
            COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
            COUNT(CASE WHEN pnl < 0 THEN 1 END) as losses,
            COUNT(*) as total,
            AVG(pnl_percent) as avg_pnl,
            SUM(pnl_percent) as total_pnl
        FROM outcomes
        WHERE DATE(date_closed) = ?
    '''
    
    df = pd.read_sql_query(query, conn, [str(date)])
    conn.close()
    
    if df.empty or df['total'][0] == 0:
        return {
            'date': date,
            'total_signals': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0
        }
    
    wins = df['wins'][0] or 0
    losses = df['losses'][0] or 0
    total = df['total'][0]
    
    return {
        'date': date,
        'total_signals': int(total),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': (wins / total * 100) if total > 0 else 0,
        'avg_pnl': df['avg_pnl'][0] or 0,
        'total_pnl': df['total_pnl'][0] or 0
    }

def get_ticker_stats(ticker):
    """Get stats for a specific ticker"""
    conn = get_db_connection()
    
    query = '''
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
            AVG(pnl_percent) as avg_return,
            COUNT(CASE WHEN hit_target = 1 THEN 1 END) as target_hits,
            COUNT(CASE WHEN hit_sl = 1 THEN 1 END) as sl_hits
        FROM outcomes
        WHERE ticker = ?
    '''
    
    df = pd.read_sql_query(query, conn, [ticker])
    conn.close()
    
    if df.empty or df['total'][0] == 0:
        return None
    
    total = df['total'][0]
    wins = df['wins'][0] or 0
    
    return {
        'ticker': ticker,
        'total_trades': int(total),
        'wins': int(wins),
        'win_rate': (wins / total * 100) if total > 0 else 0,
        'avg_return': df['avg_return'][0] or 0,
        'target_hits': int(df['target_hits'][0] or 0),
        'sl_hits': int(df['sl_hits'][0] or 0)
    }

def get_performance_timeline(days=30):
    """Get day-by-day performance"""
    conn = get_db_connection()
    
    query = '''
        SELECT 
            DATE(date_closed) as date,
            COUNT(*) as total,
            COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
            SUM(pnl_percent) as daily_return
        FROM outcomes
        WHERE date_closed >= datetime('now', '-' || ? || ' days')
        GROUP BY DATE(date_closed)
        ORDER BY date DESC
    '''
    
    df = pd.read_sql_query(query, conn, [days])
    conn.close()
    
    return df
