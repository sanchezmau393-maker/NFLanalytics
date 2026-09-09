import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

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
    
    # HistGradientBoosting es robusto y soporta NaNs nativamente
    model_home = HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=10)
    model_home.fit(X, y_home)
    
    model_away = HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=10)
    model_away.fit(X, y_away)
    
    # Predicciones dentro de muestra (para métricas de desempeño base)
    preds_home_insample = model_home.predict(X)
    preds_away_insample = model_away.predict(X)
    
    # NUEVO: Validación cruzada para calcular una varianza (Out-of-Sample) realista
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds_home_oos = cross_val_predict(model_home, X, y_home, cv=kf)
    preds_away_oos = cross_val_predict(model_away, X, y_away, cv=kf)
    
    # La desviación estándar ahora refleja el verdadero margen de error del modelo
    std_home = np.std(y_home - preds_home_oos)
    std_away = np.std(y_away - preds_away_oos)
    
    # Métricas para validación (se mantienen las de muestra para referencia rápida)
    metrics = {
        'mae_home': mean_absolute_error(y_home, preds_home_insample),
        'mae_away': mean_absolute_error(y_away, preds_away_insample),
        'rmse_home': np.sqrt(mean_squared_error(y_home, preds_home_insample)),
        'rmse_away': np.sqrt(mean_squared_error(y_away, preds_away_insample))
    }
    
    return model_home, model_away, features, std_home, std_away, metrics
