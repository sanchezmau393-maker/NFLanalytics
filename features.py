import pandas as pd
import numpy as np

def build_features(schedules):
    df = schedules.copy()
    
    # Utilizamos TODOS los juegos (jugados y futuros)
    home = df[['game_id', 'season', 'week', 'home_team', 'home_score', 'away_score']].copy()
    home.rename(columns={'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'}, inplace=True)
    home['is_home'] = 1
    
    away = df[['game_id', 'season', 'week', 'away_team', 'away_score', 'home_score']].copy()
    away.rename(columns={'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'}, inplace=True)
    away['is_home'] = 0
    
    team_games = pd.concat([home, away]).sort_values(['season', 'week'])
    
    def apply_rolling(g):
        g = g.sort_values(['season', 'week'])
        
        # Identificar solo los partidos que ya tienen un resultado real
        played_idx = g['pts_scored'].notna()
        valid_sc = g.loc[played_idx, 'pts_scored']
        valid_al = g.loc[played_idx, 'pts_allowed']
        
        # Calcular promedios SOLO sobre los partidos jugados para no diluir los datos con NaNs
        roll_17_sc = valid_sc.rolling(17, min_periods=1).mean()
        roll_3_sc = valid_sc.rolling(3, min_periods=1).mean()
        roll_5_sc = valid_sc.rolling(5, min_periods=1).mean()
        
        roll_17_al = valid_al.rolling(17, min_periods=1).mean()
        roll_3_al = valid_al.rolling(3, min_periods=1).mean()
        roll_5_al = valid_al.rolling(5, min_periods=1).mean()
        
        # Asignar los cálculos temporales respetando el índice del dataframe original
        g['temp_17_sc'] = np.nan; g.loc[played_idx, 'temp_17_sc'] = roll_17_sc
        g['temp_3_sc'] = np.nan;  g.loc[played_idx, 'temp_3_sc'] = roll_3_sc
        g['temp_5_sc'] = np.nan;  g.loc[played_idx, 'temp_5_sc'] = roll_5_sc
        
        g['temp_17_al'] = np.nan; g.loc[played_idx, 'temp_17_al'] = roll_17_al
        g['temp_3_al'] = np.nan;  g.loc[played_idx, 'temp_3_al'] = roll_3_al
        g['temp_5_al'] = np.nan;  g.loc[played_idx, 'temp_5_al'] = roll_5_al
        
        # shift(1): Mueve los datos un partido hacia adelante (Evita Data Leakage).
        # ffill(): Arrastra el último nivel conocido hacia los partidos futuros de 2026.
        g['pts_scored_season'] = g['temp_17_sc'].shift(1).ffill()
        g['pts_allowed_season'] = g['temp_17_al'].shift(1).ffill()
        g['pts_scored_l3'] = g['temp_3_sc'].shift(1).ffill()
        g['pts_allowed_l3'] = g['temp_3_al'].shift(1).ffill()
        g['pts_scored_l5'] = g['temp_5_sc'].shift(1).ffill()
        g['pts_allowed_l5'] = g['temp_5_al'].shift(1).ffill()
        
        # Limpiar columnas temporales
        g = g.drop(columns=[c for c in g.columns if c.startswith('temp_')])
        return g
        
    team_games = team_games.groupby('team', group_keys=False).apply(apply_rolling).reset_index(drop=True)
    
    # Actualización del momentum
    team_games['momentum_off'] = team_games['pts_scored_l3'] - team_games['pts_scored_season']
    team_games['momentum_def'] = team_games['pts_allowed_season'] - team_games['pts_allowed_l3']
    
    # Fillna aplica ahora SOLO para el primer partido histórico de la franquicia en 2015
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

