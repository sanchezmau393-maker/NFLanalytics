import pandas as pd
import numpy as np

def build_features(schedules):
    df = schedules.copy()
    
    df['gameday'] = pd.to_datetime(df['gameday'])
    
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
        
        # Diferencial de puntos del partido
        g.loc[played_idx, 'point_diff'] = g.loc[played_idx, 'pts_scored'] - g.loc[played_idx, 'pts_allowed']
        
        # Días de descanso desde el último partido
        g['days_since_last_game'] = g['gameday'].diff().dt.days.fillna(14).clip(upper=14)
        
        valid_sc = g.loc[played_idx, 'pts_scored']
        valid_al = g.loc[played_idx, 'pts_allowed']
        valid_diff = g.loc[played_idx, 'point_diff']
        
        # Ventanas temporales solicitadas: 3, 5, 8 y 17 semanas
        for w in [3, 5, 8, 17]:
            g[f'temp_{w}_sc'] = np.nan; g.loc[played_idx, f'temp_{w}_sc'] = valid_sc.rolling(w, min_periods=1).mean()
            g[f'temp_{w}_al'] = np.nan; g.loc[played_idx, f'temp_{w}_al'] = valid_al.rolling(w, min_periods=1).mean()
            g[f'temp_{w}_diff'] = np.nan; g.loc[played_idx, f'temp_{w}_diff'] = valid_diff.rolling(w, min_periods=1).mean()
            
            # Shift(1) estricto para prevenir Data Leakage y ffill() para partidos futuros
            g[f'pts_scored_l{w}'] = g[f'temp_{w}_sc'].shift(1).ffill()
            g[f'pts_allowed_l{w}'] = g[f'temp_{w}_al'].shift(1).ffill()
            g[f'diff_l{w}'] = g[f'temp_{w}_diff'].shift(1).ffill()

        g = g.drop(columns=[c for c in g.columns if c.startswith('temp_')])
        return g
        
    team_games = team_games.groupby('team', group_keys=False).apply(apply_rolling).reset_index(drop=True)
    
    # Momentum y Tendencias
    team_games['momentum_off'] = team_games['pts_scored_l3'] - team_games['pts_scored_l17']
    team_games['momentum_def'] = team_games['pts_allowed_l17'] - team_games['pts_allowed_l3']
    team_games['trend_diff'] = team_games['diff_l3'] - team_games['diff_l17']
    
    team_games.fillna(0, inplace=True)
    return team_games

def prepare_matchup_data(schedules, team_games):
    df = schedules.copy()
    team_games = team_games.reset_index(drop=True)
    
    cols_to_drop = ['is_home', 'pts_scored', 'pts_allowed', 'season', 'week', 'team', 'gameday', 'opponent', 'point_diff']
    
    h_feats = team_games[team_games['is_home'] == 1].copy()
    h_feats = h_feats.drop(columns=[c for c in cols_to_drop if c in h_feats.columns])
    h_feats.columns = [f"home_{c}" if c != 'game_id' else c for c in h_feats.columns]
    df = df.merge(h_feats, on='game_id', how='left')
    
    a_feats = team_games[team_games['is_home'] == 0].copy()
    a_feats = a_feats.drop(columns=[c for c in cols_to_drop if c in a_feats.columns])
    a_feats.columns = [f"away_{c}" if c != 'game_id' else c for c in a_feats.columns]
    df = df.merge(a_feats, on='game_id', how='left')
    
    return df

