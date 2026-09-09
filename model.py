import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss, accuracy_score
from sklearn.calibration import calibration_curve

def get_feature_list():
    return [
        'home_pts_scored_season', 'home_pts_allowed_season',
        'home_pts_scored_l3', 'home_pts_allowed_l3', 'home_pts_scored_l5', 'home_pts_allowed_l5', 'home_pts_scored_l8', 'home_pts_allowed_l8',
        'home_momentum_off', 'home_momentum_def', 'home_days_since_last_game', 'home_diff_l3',
        'away_pts_scored_season', 'away_pts_allowed_season',
        'away_pts_scored_l3', 'away_pts_allowed_l3', 'away_pts_scored_l5', 'away_pts_allowed_l5', 'away_pts_scored_l8', 'away_pts_allowed_l8',
        'away_momentum_off', 'away_momentum_def', 'away_days_since_last_game', 'away_diff_l3'
    ]

def train_models(matchup_df, model_type="HistGB"):
    train_df = matchup_df.dropna(subset=['home_score', 'away_score']).copy()
    features = get_feature_list()
    
    X = train_df[features].fillna(0)
    y_home = train_df['home_score']
    y_away = train_df['away_score']
    
    if model_type == "RandomForest":
        model_home = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
        model_away = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    elif model_type == "Lasso":
        model_home = Lasso(alpha=0.1, random_state=42)
        model_away = Lasso(alpha=0.1, random_state=42)
    elif model_type == "Ridge":
        model_home = Ridge(alpha=1.0, random_state=42)
        model_away = Ridge(alpha=1.0, random_state=42)
    else: # Por defecto HistGradientBoosting
        model_home = HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=10)
        model_away = HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=10)
        
    model_home.fit(X, y_home)
    model_away.fit(X, y_away)
    
    preds_home = model_home.predict(X)
    preds_away = model_away.predict(X)
    
    std_home = np.std(y_home - preds_home)
    std_away = np.std(y_away - preds_away)
    
    metrics = {
        'mae_home': mean_absolute_error(y_home, preds_home),
        'mae_away': mean_absolute_error(y_away, preds_away),
        'rmse_home': np.sqrt(mean_squared_error(y_home, preds_home)),
        'rmse_away': np.sqrt(mean_squared_error(y_away, preds_away))
    }
    
    return model_home, model_away, features, std_home, std_away, metrics

def walk_forward_validation(matchup_df):
    train_df = matchup_df.dropna(subset=['home_score', 'away_score']).copy().sort_values(['season', 'week'])
    seasons = sorted(train_df['season'].unique())
    features = get_feature_list()
    
    models = {
        'HistGB': HistGradientBoostingRegressor(random_state=42, max_iter=100),
        'RandomForest': RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }
    
    results = []
    # Validación progresiva temporal
    for i in range(2, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        
        df_train = train_df[train_df['season'].isin(train_seasons)]
        df_test = train_df[train_df['season'] == test_season]
        
        if df_test.empty: continue
        
        X_tr, yh_tr, ya_tr = df_train[features].fillna(0), df_train['home_score'], df_train['away_score']
        X_te, yh_te, ya_te = df_test[features].fillna(0), df_test['home_score'], df_test['away_score']
        
        for name, m in models.items():
            m.fit(X_tr, yh_tr); pred_h = m.predict(X_te)
            m.fit(X_tr, ya_tr); pred_a = m.predict(X_te)
            
            results.append({
                'season': test_season,
                'model': name,
                'mae': (mean_absolute_error(yh_te, pred_h) + mean_absolute_error(ya_te, pred_a)) / 2,
                'rmse': (np.sqrt(mean_squared_error(yh_te, pred_h)) + np.sqrt(mean_squared_error(ya_te, pred_a))) / 2
            })
            
    return pd.DataFrame(results)

def evaluate_win_probability(real_outcomes, predicted_probs):
    brier = brier_score_loss(real_outcomes, predicted_probs)
    logloss = log_loss(real_outcomes, predicted_probs)
    acc = accuracy_score(real_outcomes, (predicted_probs >= 0.5).astype(int))
    prob_true, prob_pred = calibration_curve(real_outcomes, predicted_probs, n_bins=5, strategy='quantile')
    return {'brier_score': brier, 'log_loss': logloss, 'accuracy': acc, 'calibration_true': prob_true, 'calibration_pred': prob_pred}
