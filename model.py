import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_models(matchup_df):
    # Entrenamos solo con partidos que ya tienen resultado
    train_df = matchup_df.dropna(subset=['home_score', 'away_score']).copy()
    
    features = [
        'home_pts_scored_season', 'home_pts_allowed_season',
        'home_pts_scored_l3', 'home_pts_allowed_l3',
        'home_momentum_off', 'home_momentum_def',
        'away_pts_scored_season', 'away_pts_allowed_season',
        'away_pts_scored_l3', 'away_pts_allowed_l3',
        'away_momentum_off', 'away_momentum_def'
    ]
    
    X = train_df[features].fillna(0)
    y_home = train_df['home_score']
    y_away = train_df['away_score']
    
    # HistGradientBoosting es robusto, soporta NaNs nativamente y es una alternativa ideal a XGBoost
    model_home = HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=10)
    model_home.fit(X, y_home)
    
    model_away = HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=10)
    model_away.fit(X, y_away)
    
    # Obtenemos predicciones sobre todo el dataset para calcular residuos
    preds_home = model_home.predict(X)
    preds_away = model_away.predict(X)
    
    # Calculamos la desviación estándar de los residuos para las simulaciones de Monte Carlo
    std_home = np.std(y_home - preds_home)
    std_away = np.std(y_away - preds_away)
    
    # Métricas para validación
    metrics = {
        'mae_home': mean_absolute_error(y_home, preds_home),
        'mae_away': mean_absolute_error(y_away, preds_away),
        'rmse_home': np.sqrt(mean_squared_error(y_home, preds_home)),
        'rmse_away': np.sqrt(mean_squared_error(y_away, preds_away))
    }
    
    return model_home, model_away, features, std_home, std_away, metrics
