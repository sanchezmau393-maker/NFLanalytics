import sqlite3
import pandas as pd
import os

DB_FILE = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            game_id TEXT PRIMARY KEY,
            season INTEGER,
            week INTEGER,
            home_team TEXT,
            away_team TEXT,
            pred_home REAL,
            pred_away REAL,
            prob_home REAL,
            prob_away REAL,
            line_used REAL,
            spread_used REAL,
            home_odds REAL DEFAULT 0.0,
            away_odds REAL DEFAULT 0.0,
            ev_home REAL DEFAULT 0.0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(game_id, season, week, home_team, away_team, pred_home, pred_away, prob_home, prob_away, line_used, spread_used, home_odds=1.9, away_odds=1.9):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Cálculo básico de Valor Esperado para el Local
    ev_home = (prob_home * home_odds) - 1
    
    cursor.execute('''
        INSERT OR REPLACE INTO predictions 
        (game_id, season, week, home_team, away_team, pred_home, pred_away, prob_home, prob_away, line_used, spread_used, home_odds, away_odds, ev_home)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (game_id, season, week, home_team, away_team, pred_home, pred_away, prob_home, prob_away, line_used, spread_used, home_odds, away_odds, ev_home))
    conn.commit()
    conn.close()

def load_predictions():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df
