import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

def walk_forward_validation(matchup_df, features):
    train_df = matchup_df.dropna(subset=['home_score', 'away_score']).copy()
    train_df = train_df.sort_values(['season', 'week'])
    
    seasons = sorted(train_df['season'].unique())
    
    models_to_test = {
        'HistGB': HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=10),
        'RandomForest': RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42)
    }
    
    results = []
    
    # Entrenar iterativamente: usar temporadas 1 a T para predecir T+1
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        
        df_train = train_df[train_df['season'].isin(train_seasons)]
        df_test = train_df[train_df['season'] == test_season]
        
        X_train, y_train_h, y_train_a = df_train[features], df_train['home_score'], df_train['away_score']
        X_test, y_test_h, y_test_a = df_test[features], df_test['home_score'], df_test['away_score']
        
        for name, model in models_to_test.items():
            # Modelo Local
            model.fit(X_train, y_train_h)
            pred_h = model.predict(X_test)
            
            # Modelo Visitante
            model.fit(X_train, y_train_a)
            pred_a = model.predict(X_test)
            
            mae_h = mean_absolute_error(y_test_h, pred_h)
            mae_a = mean_absolute_error(y_test_a, pred_a)
            
            results.append({
                'season_tested': test_season,
                'model': name,
                'mae_home': mae_h,
                'mae_away': mae_a,
                'mae_total': (mae_h + mae_a) / 2
            })
            
    return pd.DataFrame(results)
