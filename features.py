import pandas as pd
import numpy as np

def build_features(schedules):
    df = schedules.copy()
    # Filtramos para construir el histórico solo con juegos finalizados
    df = df.dropna(subset=['home_score', 'away_score'])
    
    # Vista equipo local
    home = df[['game_id', 'season', 'week', 'home_team', 'home_score', 'away_score']].copy()
    home.rename(columns={'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'}, inplace=True)
    home['is_home'] = 1
    
    # Vista equipo visitante
    away = df[['game_id', 'season', 'week', 'away_team', 'away_score', 'home_score']].copy()
    away.rename(columns={'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'}, inplace=True)
    away['is_home'] = 0
    
    team_games = pd.concat([home, away]).sort_values(['season', 'week'])
    
    def roll_stats(g):
        g = g.sort_values('week')
        # SHIFT(1) ES OBLIGATORIO: previene data leakage garantizando que solo usamos datos ANTERIORES al partido
        g['pts_scored_season'] = g['pts_scored'].shift(1).expanding().mean()
        g['pts_allowed_season'] = g['pts_allowed'].shift(1).expanding().mean()
        
        g['pts_scored_l3'] = g['pts_scored'].shift(1).rolling(3, min_periods=1).mean()
        g['pts_allowed_l3'] = g['pts_allowed'].shift(1).rolling(3, min_periods=1).mean()
        
        g['pts_scored_l5'] = g['pts_scored'].shift(1).rolling(5, min_periods=1).mean()
        g['pts_allowed_l5'] = g['pts_allowed'].shift(1).rolling(5, min_periods=1).mean()
        return g
        
    # Agrupamos y reseteamos el índice para evitar que las columnas desaparezcan y causen un KeyError
    team_games = team_games.groupby(['season', 'team'], group_keys=False).apply(roll_stats).reset_index(drop=True)
    
    # Cálculo de momentum (Tendencia reciente vs promedio de temporada)
    team_games['momentum_off'] = team_games['pts_scored_l3'] - team_games['pts_scored_season']
    team_games['momentum_def'] = team_games['pts_allowed_season'] - team_games['pts_allowed_l3']
    
    # Valores por defecto para semanas iniciales
    team_games.fillna({
        'pts_scored_season': 21.0, 'pts_allowed_season': 21.0,
        'pts_scored_l3': 21.0, 'pts_allowed_l3': 21.0,
        'pts_scored_l5': 21.0, 'pts_allowed_l5': 21.0,
        'momentum_off': 0.0, 'momentum_def': 0.0
    }, inplace=True)
    
    return team_games

def prepare_matchup_data(schedules, team_games):
    df = schedules.copy()
    
    # Aseguramos que el índice esté limpio antes de cruzar datos
    team_games = team_games.reset_index(drop=True)
    
    # --- DATOS DEL EQUIPO LOCAL ---
    h_feats = team_games[team_games['is_home'] == 1].copy()
    
    # Eliminamos columnas innecesarias para evitar cruces erróneos
    cols_to_drop = ['is_home', 'pts_scored', 'pts_allowed', 'season', 'week', 'team']
    h_feats = h_feats.drop(columns=[c for c in cols_to_drop if c in h_feats.columns])
    
    # Renombramos añadiendo 'home_' a todo excepto al game_id (llave principal)
    h_feats.columns = [f"home_{c}" if c != 'game_id' else c for c in h_feats.columns]
    
    # Unimos usando SOLO el game_id (identificador único del partido)
    df = df.merge(h_feats, on='game_id', how='left')
    
    # --- DATOS DEL EQUIPO VISITANTE ---
    a_feats = team_games[team_games['is_home'] == 0].copy()
    a_feats = a_feats.drop(columns=[c for c in cols_to_drop if c in a_feats.columns])
    
    a_feats.columns = [f"away_{c}" if c != 'game_id' else c for c in a_feats.columns]
    df = df.merge(a_feats, on='game_id', how='left')
    
    return df
