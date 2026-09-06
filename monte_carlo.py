import numpy as np

def run_simulation(exp_home, exp_away, std_home, std_away, n_sims=10000, seed=42):
    rng = np.random.default_rng(seed)
    
    # Simular puntuaciones basadas en la predicción y el error residual del modelo
    sim_home = rng.normal(exp_home, std_home, n_sims)
    sim_away = rng.normal(exp_away, std_away, n_sims)
    
    # Las puntuaciones no pueden ser negativas
    sim_home = np.maximum(sim_home, 0)
    sim_away = np.maximum(sim_away, 0)
    
    diff = sim_home - sim_away
    total = sim_home + sim_away
    
    prob_home_win = np.mean(diff > 0)
    prob_away_win = np.mean(diff < 0)
    prob_tie = np.mean(np.round(diff) == 0)
    
    return {
        'sim_home': sim_home,
        'sim_away': sim_away,
        'diff': diff,
        'total': total,
        'prob_home': prob_home_win,
        'prob_away': prob_away_win,
        'prob_tie': prob_tie,
        'med_total': np.median(total),
        'med_diff': np.median(diff),
        'pct_95_total': np.percentile(total, 95),
        'pct_05_total': np.percentile(total, 5)
    }
