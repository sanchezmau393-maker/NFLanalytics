import numpy as np

def run_simulation(mu_home, mu_away, std_home, std_away, n_sims=10000, rho=0.15):
    cov = rho * std_home * std_away
    cov_matrix = [[std_home**2, cov], [cov, std_away**2]]
    
    sims = np.random.multivariate_normal([mu_home, mu_away], cov_matrix, n_sims)
    
    sim_home = np.maximum(0, np.round(sims[:, 0]))
    sim_away = np.maximum(0, np.round(sims[:, 1]))
    
    diff = sim_home - sim_away
    total = sim_home + sim_away
    
    return {
        'sim_home': sim_home, 'sim_away': sim_away, 'diff': diff, 'total': total,
        'prob_home': np.mean(diff > 0), 'prob_away': np.mean(diff < 0), 'prob_tie': np.mean(diff == 0)
    }
