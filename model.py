import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

# Suprimir advertencias de convergencia para modelos lineales en validación rápida
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def select_best_model_temporal(X, y, models, n_splits=5):
    """
    Evalúa un diccionario de modelos utilizando validación cruzada temporal.
    Garantiza que ninguna predicción utilice datos del futuro.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_model_name = None
    best_mae = float('inf')
    model_results = {}

    for name, model in models.items():
        maes = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
        
        avg_mae = np.mean(maes)
        model_results[name] = avg_mae
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_model_name = name
            
    return best_model_name, best_mae, model_results

def train_models(matchups):
    """
    Ordena cronológicamente los datos, evalúa los modelos secuencialmente, 
    selecciona el mejor basado en MAE fuera de muestra y entrena los modelos finales.
    """
    # 1. ORDEN TEMPORAL ESTRICTO (El núcleo para evitar data leakage)
    played_games = matchups.dropna(subset=['home_score', 'away_score']).copy()
    played_games = played_games.sort_values(['season', 'week']).reset_index(drop=True)
    
    # 2. SEPARACIÓN DE VARIABLES
    exclude_cols = [
        'game_id', 'season', 'week', 'date', 'home_team', 'away_team', 
        'home_score', 'away_score', 'total_line', 'spread_line', 
        'home_moneyline', 'away_moneyline'
    ]
    
    feat_cols = [c for c in played_games.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(played_games[c])]
    
    X = played_games[feat_cols].fillna(0)
    y_home = played_games['home_score']
    y_away = played_games['away_score']
    
    # 3. SUITE DE MODELOS A EVALUAR
    models_to_evaluate = {
        'Ridge': Ridge(alpha=5.0, random_state=42),
        'Lasso': Lasso(alpha=0.5, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    }
    
    if HAS_XGB:
        models_to_evaluate['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        
    # 4. SELECCIÓN OUT-OF-SAMPLE PARA EQUIPO LOCAL
    best_name_h, best_mae_h, results_h = select_best_model_temporal(X, y_home, models_to_evaluate)
    best_model_home = models_to_evaluate[best_name_h]
    best_model_home.fit(X, y_home)
    
    # 5. SELECCIÓN OUT-OF-SAMPLE PARA EQUIPO VISITANTE
    best_name_a, best_mae_a, results_a = select_best_model_temporal(X, y_away, models_to_evaluate)
    best_model_away = models_to_evaluate[best_name_a]
    best_model_away.fit(X, y_away)
    
    # 6. CÁLCULO REALISTA DE DESVIACIÓN ESTÁNDAR PARA MONTE CARLO
    std_home = max(3.0, best_mae_h * 1.253)
    std_away = max(3.0, best_mae_a * 1.253)
    
    metrics = {
        'best_model_home': best_name_h,
        'mae_home': best_mae_h,
        'best_model_away': best_name_a,
        'mae_away': best_mae_a,
        'results_home': results_h,
        'results_away': results_a
    }
    
    return best_model_home, best_model_away, feat_cols, std_home, std_away, metrics
