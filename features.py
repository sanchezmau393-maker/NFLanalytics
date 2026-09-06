import pandas as pd
import numpy as np

def build_features(schedules):
    df = schedules.copy()
    played = df.dropna(subset=['home_score', 'away_score']).copy()
    
    # Vista local
    home = played[['game_id', 'season', 'week', 'home_team', 'home_score', 'away_score']].copy()
    home.rename(columns={'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'}, inplace=True)
    home['is_home'] = 1
    
    # Vista visitante
    away = played[['game_id', 'season', 'week', 'away_team', 'away_score', 'home_score']].copy()
    away.rename(columns={'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'}, inplace=True)
    away['is_home'] = 0
    
    team_games = pd.concat([home, away]).sort_values(['season', 'week'])
    
    def roll_stats(g):
        g = g.sort_values('week')
        # SHIFT(1) OBLIGATORIO: previene Data Leakage usará solo datos PREVIOS al juego
        g['pts_scored_season'] = g['pts_scored'].shift(1).expanding().mean()
        g['pts_allowed_season'] = g['pts_allowed'].shift(1).expanding().mean()
        
        g['pts_scored_l3'] = g['pts_scored'].shift(1).rolling(3, min_periods=1).mean()
        g['pts_allowed_l3'] = g['pts_allowed'].shift(1).rolling(3, min_periods=1).mean()
        
        g['pts_scored_l5'] = g['pts_scored'].shift(1).rolling(5, min_periods=1).mean()
        g['pts_allowed_l5'] = g['pts_allowed'].shift(1).rolling(5, min_periods=1).mean()
        return g
        
    team_games = team_games.groupby(['season', 'team'], group_keys=False).apply(roll_stats).reset_index(drop=True)
    
    team_games['momentum_off'] = team_games['pts_scored_l3'] - team_games['pts_scored_season']
    team_games['momentum_def'] = team_games['pts_allowed_season'] - team_games['pts_allowed_l3']
    
    team_games.fillna({
        'pts_scored_season': 21.0, 'pts_allowed_season': 21.0,
        'pts_scored_l3': 21.0, 'pts_allowed_l3': 21.0,
        'pts_scored_l5': 21.0, 'pts_allowed_l5': 21.0,
        'momentum_off': 0.0, 'momentum_def': 0.0
    }, inplace=True)
    
    return team_games

def prepare_matchup_data(schedules, team_games):
    df = schedules.copy()
    team_games = team_games.reset_index(drop=True)
    
    cols_to_drop = ['is_home', 'pts_scored', 'pts_allowed', 'season', 'week', 'team']
    
    # Merge Local
    h_feats = team_games[team_games['is_home'] == 1].copy()
    h_feats = h_feats.drop(columns=[c for c in cols_to_drop if c in h_feats.columns])
    h_feats.columns = [f"home_{c}" if c != 'game_id' else c for c in h_feats.columns]
    df = df.merge(h_feats, on='game_id', how='left')
    
    # Merge Visitante
    a_feats = team_games[team_games['is_home'] == 0].copy()
    a_feats = a_feats.drop(columns=[c for c in cols_to_drop if c in a_feats.columns])
    a_feats.columns = [f"away_{c}" if c != 'game_id' else c for c in a_feats.columns]
    df = df.merge(a_feats, on='game_id', how='left')
    
    return df
