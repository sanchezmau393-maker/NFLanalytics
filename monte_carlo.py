import numpy as np

def run_simulation(pred_home, pred_away, std_home, std_away, n_sims=10000, correlation=0.15):
    """
    Simulación Monte Carlo usando Distribución Normal Bivariada para capturar 
    la dependencia entre la ofensiva local y visitante (pace of play).
    """
    # Evitar desviaciones estándar en cero
    std_home = max(0.1, std_home)
    std_away = max(0.1, std_away)
    
    # Matriz de covarianza
    cov = correlation * std_home * std_away
    cov_matrix = [[std_home**2, cov], [cov, std_away**2]]
    
    # Generar simulaciones conjuntas
    sims = np.random.multivariate_normal([pred_home, pred_away], cov_matrix, n_sims)
    
    # Truncar a 0 (no existen puntos negativos en la NFL)
    sims_home = np.maximum(0, sims[:, 0])
    sims_away = np.maximum(0, sims[:, 1])
    
    # Convención estricta: Spread = Visitante - Local (Spread negativo = Local Favorito)
    diff = sims_away - sims_home  
    total = sims_home + sims_away
    
    prob_home = np.mean(sims_home > sims_away)
    prob_away = np.mean(sims_away > sims_home)
    prob_tie = np.mean(np.round(sims_home) == np.round(sims_away))
    
    return {
        'prob_home': prob_home,
        'prob_away': prob_away,
        'prob_tie': prob_tie,
        'sims_home': sims_home,
        'sims_away': sims_away,
        'diff': diff,
        'total': total
    }
