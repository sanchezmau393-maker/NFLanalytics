import pandas as pd
import numpy as np

def build_features(schedules):
    df = schedules.copy()
    
    # Asegurar que gameday sea formato fecha para calcular descansos
    df['gameday'] = pd.to_datetime(df['gameday'])
    
    # Separar local y visitante
    home = df[['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home.rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'}, inplace=True)
    home['is_home'] = 1
    
    away = df[['game_id', 'season', 'week', 'gameday', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away.rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'}, inplace=True)
    away['is_home'] = 0
    
    team_games = pd.concat([home, away]).sort_values(['team', 'season', 'week'])
    
    def apply_rolling(g):
        g = g.sort_values(['season', 'week'])
        played_idx = g['pts_scored'].notna()
        
        # Diferencial de puntos
        g.loc[played_idx, 'point_diff'] = g.loc[played_idx, 'pts_scored'] - g.loc[played_idx, 'pts_allowed']
        
        # Días de descanso (diferencia entre el gameday actual y el anterior)
        g['days_since_last_game'] = g['gameday'].diff().dt.days
        # Llenar el primer partido de la temporada asumiendo descanso completo (ej. 14 días o más)
        g['days_since_last_game'] = g['days_since_last_game'].fillna(14).clip(upper=14) 
        
        valid_sc = g.loc[played_idx, 'pts_scored']
        valid_al = g.loc[played_idx, 'pts_allowed']
        valid_diff = g.loc[played_idx, 'point_diff']
        
        # Promedios móviles (3, 5, 8, 17)
        for w in [3, 5, 8, 17]:
            g[f'temp_{w}_sc'] = np.nan; g.loc[played_idx, f'temp_{w}_sc'] = valid_sc.rolling(w, min_periods=1).mean()
            g[f'temp_{w}_al'] = np.nan; g.loc[played_idx, f'temp_{w}_al'] = valid_al.rolling(w, min_periods=1).mean()
            g[f'temp_{w}_diff'] = np.nan; g.loc[played_idx, f'temp_{w}_diff'] = valid_diff.rolling(w, min_periods=1).mean()
            
            # Aplicar shift(1) para evitar Data Leakage
            g[f'pts_scored_l{w}'] = g[f'temp_{w}_sc'].shift(1).ffill()
            g[f'pts_allowed_l{w}'] = g[f'temp_{w}_al'].shift(1).ffill()
            g[f'diff_l{w}'] = g[f'temp_{w}_diff'].shift(1).ffill()

        # Limpieza de temporales
        g = g.drop(columns=[c for c in g.columns if c.startswith('temp_')])
        return g
        
    team_games = team_games.groupby('team', group_keys=False).apply(apply_rolling).reset_index(drop=True)
    
    # Eficiencia y Tendencias (Momentum)
    team_games['momentum_off_3_17'] = team_games['pts_scored_l3'] - team_games['pts_scored_l17']
    team_games['momentum_def_3_17'] = team_games['pts_allowed_l17'] - team_games['pts_allowed_l3']
    team_games['trend_diff'] = team_games['diff_l3'] - team_games['diff_l17']
    
    team_games.fillna(0, inplace=True) # Manejo de NaNs iniciales
    return team_games


