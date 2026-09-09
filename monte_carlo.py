import numpy as np
import pandas as pd

def run_simulation(mu_home, mu_away, std_home, std_away, n_sims=10000, rho=0.15):
    """
    Simulación Monte Carlo utilizando Normal Bivariada para capturar
    la correlación entre los puntos del Local y el Visitante.
    rho=0.15 asume que los partidos de alto ritmo benefician a ambos.
    """
    # Matriz de covarianza
    cov = rho * std_home * std_away
    cov_matrix = [[std_home**2, cov], [cov, std_away**2]]
    
    # Simular
    sims = np.random.multivariate_normal([mu_home, mu_away], cov_matrix, n_sims)
    
    # Los puntos en NFL no son negativos y son enteros
    sim_home = np.maximum(0, np.round(sims[:, 0]))
    sim_away = np.maximum(0, np.round(sims[:, 1]))
    
    diff = sim_home - sim_away
    total = sim_home + sim_away
    
    # Calcular probabilidades base
    prob_home = np.mean(diff > 0)
    prob_away = np.mean(diff < 0)
    prob_tie = np.mean(diff == 0)
    
    return {
        'sim_home': sim_home,
        'sim_away': sim_away,
        'diff': diff,
        'total': total,
        'prob_home': prob_home,
        'prob_away': prob_away,
        'prob_tie': prob_tie
    }
