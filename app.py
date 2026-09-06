import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import itertools
from operator import itemgetter
import data_loader
import features
import model
import monte_carlo
import tracker

st.set_page_config(page_title="NFL Analytics Pro", page_icon="🏈", layout="wide")

if 'selected_game_id' not in st.session_state:
    st.session_state.selected_game_id = None

def select_game(game_id):
    st.session_state.selected_game_id = game_id

def get_ml_odds(american_odds, prob):
    """Convierte momio americano a decimal o calcula cuota justa si no hay momio de Vegas."""
    if pd.notna(american_odds) and american_odds != "" and american_odds != 0:
        try:
            odds = float(american_odds)
            if odds > 0: return (odds / 100.0) + 1.0
            if odds < 0: return (100.0 / abs(odds)) + 1.0
        except: pass
    # Fallback: Cuota justa calculada por el modelo si la casa de apuestas aún no abre la línea
    return max(1.05, 1.0 / prob) if prob > 0 else 1.05

@st.cache_data(show_spinner=False)
def preparar_datos_y_momentum(schedules):
    t_games = features.build_features(schedules)
    return features.prepare_matchup_data(schedules, t_games)

@st.cache_resource(show_spinner=False)
def entrenar_modelos_ml(matchups):
    return model.train_models(matchups)

st.title("🏈 NFL Analytics Pro")
st.markdown("Análisis avanzado, predicción objetiva de bajas y Player Props mediante Machine Learning y Monte Carlo.")

# --- 1. CARGA DE DATOS ---
with st.spinner("Descargando calendarios e históricos de la NFL..."):
    schedules = data_loader.load_historical_schedules(2015, 2026)
    if schedules.empty:
        st.error("Error al obtener los calendarios. Por favor recarga la página.")
        st.stop()

# --- 2. INGENIERÍA DE CARACTERÍSTICAS Y ENTRENAMIENTO (CON PRECARGA) ---
with st.spinner("Cargando momentum y modelos predictivos (solo toma unos segundos la primera vez)..."):
    matchups = preparar_datos_y_momentum(schedules)
    m_home, m_away, feat_cols, std_home, std_away, metrics = entrenar_modelos_ml(matchups)

seasons_in_data = sorted(schedules['season'].dropna().unique(), reverse=True)
recent_years = [int(seasons_in_data[0])]
if len(seasons_in_data) > 1:
    recent_years.append(int(seasons_in_data[1]))

weekly_data = data_loader.load_weekly_data(recent_years)

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración")
selected_season = st.sidebar.selectbox("Temporada:", seasons_in_data, index=0)

season_data = matchups[matchups['season'] == selected_season]
weeks_avail = sorted([int(w) for w in season_data['week'].dropna().unique()])

unplayed = season_data[season_data['home_score'].isna()]
default_week = int(unplayed['week'].min()) if not unplayed.empty else (weeks_avail[0] if weeks_avail else 1)

selected_week = st.sidebar.selectbox("Semana:", weeks_avail, index=weeks_avail.index(default_week) if default_week in weeks_avail else 0)

# CAMBIO 1: Se agrega "🐛 Diagnóstico" a la lista de pestañas
tabs = st.tabs([
    "📅 Cartelera", 
    "🔮 Predicción", 
    "🎯 Eficiencia",
    "🏃 Player Props",
    "🔗 Combinadas",
    "📊 Registro", 
    "📈 Info Modelo",
    "🐛 Diagnóstico" 
])

week_games = season_data[season_data['week'] == selected_week]

# --- PESTAÑA 1: CARTELERA COMPLETA ---
with tabs[0]:
    st.header(f"Todos los Partidos - Semana {selected_week} ({selected_season})")
    if week_games.empty:
        st.warning("No hay partidos registrados para esta semana.")
    else:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(week_games.iterrows()):
            col = cols[idx % 3]
            with col:
                st.container(border=True)
                st.markdown(f"### {row['away_team']} @ {row['home_team']}")
                if pd.notna(row['home_score']) and pd.notna(row['away_score']):
                    st.write(f"**Resultado:** {row['away_team']} {int(row['away_score'])} - {int(row['home_score'])} {row['home_team']}")
                else:
                    st.write("⏳ *Partido Pendiente*")
                
                if st.button(f"Analizar {row['away_team']} vs {row['home_team']}", key=f"btn_{row['game_id']}", use_container_width=True):
                    select_game(row['game_id'])
                    st.success("¡Partido seleccionado! Pasa a la pestaña '🔮 Predicción'.")

# --- PESTAÑA 2: PREDICCIÓN ---
with tabs[1]:
    if not st.session_state.selected_game_id:
        st.info("👈 Por favor selecciona un partido desde la pestaña '📅 Cartelera'.")
    else:
        g = matchups[matchups['game_id'] == st.session_state.selected_game_id]
        if g.empty:
            st.error("Partido no encontrado.")
        else:
            g = g.iloc[0]
            st.header(f"Análisis: {g['away_team']} @ {g['home_team']}")
            
            st.subheader("🚑 Reporte Automático de Lesiones y Bajas")
            col_inj1, col_inj2 = st.columns(2)
            
            roster_home = sorted(weekly_data[weekly_data['recent_team'] == g['home_team']]['player_display_name'].dropna().unique()) if not weekly_data.empty else []
            roster_away = sorted(weekly_data[weekly_data['recent_team'] == g['away_team']]['player_display_name'].dropna().unique()) if not weekly_data.empty else []
            
            with col_inj1:
                bajas_home = st.multiselect(f"Bajas en {g['home_team']}:", roster_home)
            with col_inj2:
                bajas_away = st.multiselect(f"Bajas en {g['away_team']}:", roster_away)
                
            penal_home = 0.0
            for player in bajas_home:
                pos = weekly_data[weekly_data['player_display_name'] == player]['position'].iloc[0] if not weekly_data.empty else 'DEF'
                if pos == 'QB': penal_home += 4.0
                elif pos in ['WR', 'RB', 'TE']: penal_home += 1.5
                else: penal_home += 0.5
                
            penal_away = 0.0
            for player in bajas_away:
                pos = weekly_data[weekly_data['player_display_name'] == player]['position'].iloc[0] if not weekly_data.empty else 'DEF'
                if pos == 'QB': penal_away += 4.0
                elif pos in ['WR', 'RB', 'TE']: penal_away += 1.5
                else: penal_away += 0.5
            
            if penal_home > 0 or penal_away > 0:
                st.warning(f"Descuento automático por bajas: -{penal_home:.1f} pts a {g['home_team']} | -{penal_away:.1f} pts a {g['away_team']}")

            col_m1, col_m2 = st.columns(2)
            adj_manual_h = col_m1.number_input(f"Ajuste manual extra {g['home_team']} (Pts):", value=0.0, step=0.5)
            adj_manual_a = col_m2.number_input(f"Ajuste manual extra {g['away_team']} (Pts):", value=0.0, step=0.5)

            X_match = g[feat_cols].to_frame().T.fillna(0)
            pred_h_base = max(0, m_home.predict(X_match)[0])
            pred_a_base = max(0, m_away.predict(X_match)[0])
            
            pred_h = max(0, pred_h_base - penal_home + adj_manual_h)
            pred_a = max(0, pred_a_base - penal_away + adj_manual_a)

            st.markdown("---")
            st.subheader("🎲 Simulaciones Monte Carlo")
            col_L1, col_L2, col_L3 = st.columns(3)
            
            auto_ou = float(g['total_line']) if pd.notna(g['total_line']) else 45.5
            auto_spread = float(g['spread_line']) if pd.notna(g['spread_line']) else 0.0
            
            ou_line = col_L1.number_input("Línea Over/Under:", value=auto_ou)
            spread_line = col_L2.number_input("Spread Local:", value=auto_spread)
            n_sims = col_L3.selectbox("Simulaciones:", [1000, 10000, 50000], index=1)

            if st.button("🚀 Ejecutar Predicción", type="primary"):
                res = monte_carlo.run_simulation(pred_h, pred_a, std_home, std_away, n_sims=n_sims)
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Prob. {g['home_team']}", f"{res['prob_home']*100:.1f}%")
                c2.metric("Empate", f"{res['prob_tie']*100:.1f}%")
                c3.metric(f"Prob. {g['away_team']}", f"{res['prob_away']*100:.1f}%")
                
                st.write(f"### 🎯 Marcador Esperado: {g['home_team']} {pred_h:.1f} - {pred_a:.1f} {g['away_team']}")
                
                prob_over = np.mean(res['total'] > ou_line)
                prob_under = np.mean(res['total'] < ou_line)
                prob_cover = np.mean(res['diff'] > spread_line)
                
                st.markdown(f"**Over {ou_line}:** {prob_over*100:.1f}% | **Under {ou_line}:** {prob_under*100:.1f}%")
                st.markdown(f"**Probabilidad {g['home_team']} cubre Spread ({spread_line}):** {prob_cover*100:.1f}%")
                
                tracker.save_prediction(
                    g['game_id'], g['season'], g['week'], g['home_team'], g['away_team'],
                    pred_h, pred_a, res['prob_home'], res['prob_away'], ou_line, spread_line
                )
                st.success("✅ Predicción registrada en el historial.")

# --- PESTAÑA 3: EFICIENCIA DE PREDICCIÓN ---
with tabs[2]:
    st.header(f"🎯 Eficiencia de Predicción - Temporada {selected_season}")
    
    played_games = season_data[season_data['home_score'].notna()].copy()
    if played_games.empty:
        st.info(f"Aún no hay resultados finales para evaluar en la temporada {selected_season}.")
    else:
        X_hist = played_games[feat_cols].fillna(0)
        played_games['pred_home'] = m_home.predict(X_hist)
        played_games['pred_away'] = m_away.predict(X_hist)
        played_games['real_winner'] = np.where(played_games['home_score'] > played_games['away_score'], played_games['home_team'], played_games['away_team'])
        played_games['pred_winner'] = np.where(played_games['pred_home'] > played_games['pred_away'], played_games['home_team'], played_games['away_team'])
        played_games['is_correct'] = played_games['real_winner'] == played_games['pred_winner']
        valid_games = played_games[played_games['home_score'] != played_games['away_score']]
        
        total_valid = len(valid_games)
        correct_preds = valid_games['is_correct'].sum()
        acc_percentage = (correct_preds / total_valid * 100) if total_valid > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos Evaluados", total_valid)
        c2.metric("Predicciones Acertadas", correct_preds)
        c3.metric("Eficacia General (%)", f"{acc_percentage:.1f}%")
        
        st.subheader("Evolución de Eficacia por Semana")
        weekly_acc = valid_games.groupby('week')['is_correct'].mean() * 100
        st.line_chart(weekly_acc)

# --- PESTAÑA 4: PLAYER PROPS ---
with tabs[3]:
    st.header("🏃 Player Props (Probabilidades por Jugador)")
    if weekly_data.empty:
        st.warning("No hay datos de jugadores disponibles actualmente.")
    else:
        teams_avail = sorted(weekly_data['recent_team'].dropna().unique())
        col_p1, col_p2 = st.columns(2)
        selected_team = col_p1.selectbox("Selecciona Equipo:", teams_avail)
        team_players = weekly_data[weekly_data['recent_team'] == selected_team]
        player_names = sorted(team_players['player_display_name'].dropna().unique())
        selected_player = col_p2.selectbox("Selecciona Jugador:", player_names)
        
        p_data = team_players[team_players['player_display_name'] == selected_player]
        if not p_data.empty:
            pos = p_data['position'].iloc[0]
            st.markdown(f"### Análisis de {selected_player} ({pos})")
            
            metrics_list = []
            if pos == 'QB': metrics_list = [('passing_yards', 'Yardas por Pase'), ('attempts', 'Intentos de Pase')]
            elif pos in ['WR', 'TE']: metrics_list = [('receiving_yards', 'Yardas por Recepción'), ('receptions', 'Recepciones')]
            elif pos == 'RB': metrics_list = [('rushing_yards', 'Yardas Terrestres'), ('carries', 'Acarreos')]
            else: metrics_list = [('receiving_yards', 'Yardas Totales'), ('receptions', 'Recepciones')]

            probs = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
            labels = ["90% (Muy Seguro)", "80%", "70%", "60%", "50% (Promedio)", "40%", "30%", "20%", "10% (Arriesgado)"]
            df_props = pd.DataFrame({"Probabilidad (OVER)": labels})
            
            for col_stat, stat_name in metrics_list:
                stat_values = p_data[col_stat].dropna()
                if len(stat_values) >= 2:
                    mu, sigma = stat_values.mean(), stat_values.std()
                    sigma = 0.1 if sigma == 0 or np.isnan(sigma) else sigma
                    lines = [max(0, np.round(norm.ppf(1 - p, loc=mu, scale=sigma), 1)) for p in probs]
                    df_props[f"Línea de {stat_name}"] = lines
                    st.caption(f"**{stat_name}:** Promedio {mu:.1f} | Desviación {sigma:.1f}")
            st.dataframe(df_props, use_container_width=True, hide_index=True)

# --- PESTAÑA 5: COMBINADAS INTELIGENTES ---
with tabs[4]:
    st.header("🔗 Generador Inteligente de Combinadas (Frontera Eficiente)")
    
    if week_games.empty:
        st.info("No hay partidos pendientes para analizar en esta semana.")
    else:
        st.write("Selecciona tu nivel de riesgo. El algoritmo calculará todas las combinaciones posibles de la semana actual respetando las reglas de Las Vegas (no ML + Spread del mismo juego) y te ofrecerá aquellas que tienen el mayor beneficio esperado (Expected Value).")
        
        c1, c2 = st.columns(2)
        riesgo = c1.selectbox("Nivel de Riesgo (Prob. de éxito de la combinada):", [
            "Muy Conservadora (Prob. 25% - 40%)",
            "Equilibrada (Prob. 10% - 25%)",
            "Arriesgada (Prob. 3% - 10%)",
            "Lotería (Prob. < 3%)"
        ])
        
        lista_juegos = ["Ninguno"] + [f"{r['away_team']} @ {r['home_team']}" for _, r in week_games.iterrows()]
        juego_obligatorio = c2.selectbox("Incluir un partido obligatoriamente:", lista_juegos)
        
        if st.button("🚀 Generar Opciones de Combinadas", type="primary"):
            with st.spinner("Simulando miles de combinaciones..."):
                all_picks = []
                # 1. Extraer probabilidades de todos los juegos de la semana
                for _, row in week_games.iterrows():
                    match_data = matchups[matchups['game_id'] == row['game_id']].iloc[0]
                    X_m = match_data[feat_cols].to_frame().T.fillna(0)
                    p_h = max(0, m_home.predict(X_m)[0])
                    p_a = max(0, m_away.predict(X_m)[0])
                    
                    res = monte_carlo.run_simulation(p_h, p_a, std_home, std_away, n_sims=2000)
                    
                    ou_line = float(row['total_line']) if pd.notna(row['total_line']) else 45.5
                    spread_line = float(row['spread_line']) if pd.notna(row['spread_line']) else 0.0
                    
                    p_over = np.mean(res['total'] > ou_line)
                    p_under = np.mean(res['total'] < ou_line)
                    p_cov_h = np.mean(res['diff'] > spread_line)
                    p_cov_a = np.mean(res['diff'] < spread_line)
                    
                    gid = row['game_id']
                    m_str = f"{row['away_team']} @ {row['home_team']}"
                    
                    all_picks.append({'id': f"{gid}_ML_H", 'game': gid, 'match': m_str, 'type': 'ML', 'desc': f"Gana {row['home_team']}", 'prob': res['prob_home'], 'odds': get_ml_odds(row.get('home_moneyline'), res['prob_home'])})
                    all_picks.append({'id': f"{gid}_ML_A", 'game': gid, 'match': m_str, 'type': 'ML', 'desc': f"Gana {row['away_team']}", 'prob': res['prob_away'], 'odds': get_ml_odds(row.get('away_moneyline'), res['prob_away'])})
                    all_picks.append({'id': f"{gid}_SP_H", 'game': gid, 'match': m_str, 'type': 'Spread', 'desc': f"{row['home_team']} cubre Spread ({spread_line})", 'prob': p_cov_h, 'odds': 1.91})
                    all_picks.append({'id': f"{gid}_SP_A", 'game': gid, 'match': m_str, 'type': 'Spread', 'desc': f"{row['away_team']} cubre Spread ({-spread_line})", 'prob': p_cov_a, 'odds': 1.91})
                    all_picks.append({'id': f"{gid}_OU_O", 'game': gid, 'match': m_str, 'type': 'OU', 'desc': f"OVER {ou_line}", 'prob': p_over, 'odds': 1.91})
                    all_picks.append({'id': f"{gid}_OU_U", 'game': gid, 'match': m_str, 'type': 'OU', 'desc': f"UNDER {ou_line}", 'prob': p_under, 'odds': 1.91})
                
                valid_parlays = []
                # 2. Generar combinaciones de 2 y 3 selecciones
                for k in [2, 3]:
                    for combo in itertools.combinations(all_picks, k):
                        if juego_obligatorio != "Ninguno" and juego_obligatorio not in [p['match'] for p in combo]:
                            continue
                            
                        # Validación de Reglas: Máximo 1 ML/Spread por juego (Evitar conflicto) y máximo 1 O/U por juego
                        is_valid = True
                        g_types = {}
                        for p in combo:
                            if p['game'] not in g_types: g_types[p['game']] = []
                            g_types[p['game']].append(p['type'])
                            
                        for g_id, t_list in g_types.items():
                            if 'ML' in t_list and 'Spread' in t_list: is_valid = False; break
                            if t_list.count('ML') > 1 or t_list.count('Spread') > 1 or t_list.count('OU') > 1: is_valid = False; break
                            
                        if not is_valid: continue
                        
                        c_prob = np.prod([p['prob'] for p in combo])
                        c_odds = np.prod([p['odds'] for p in combo])
                        c_ev = (c_prob * c_odds) - 1 # Expected Value
                        
                        # Filtro de Riesgo
                        if "Conservadora" in riesgo and not (0.25 <= c_prob <= 0.40): continue
                        if "Equilibrada" in riesgo and not (0.10 <= c_prob < 0.25): continue
                        if "Arriesgada" in riesgo and not (0.03 <= c_prob < 0.10): continue
                        if "Lotería" in riesgo and c_prob >= 0.03: continue
                        
                        valid_parlays.append({'combo': combo, 'prob': c_prob, 'odds': c_odds, 'ev': c_ev})

                if not valid_parlays:
                    st.warning("No se encontraron combinadas que cumplan estrictamente con ese nivel de riesgo y reglas. Intenta cambiar el riesgo.")
                else:
                    # Ordenar por el mejor Expected Value (El mejor beneficio en relación a su riesgo)
                    valid_parlays.sort(key=itemgetter('ev'), reverse=True)
                    top_parlays = valid_parlays[:8]
                    
                    st.success(f"Se generaron las mejores {len(top_parlays)} opciones optimizadas:")
                    for i, parl in enumerate(top_parlays):
                        with st.expander(f"⭐ Opción {i+1} | Probabilidad: {parl['prob']*100:.1f}% | Pago: x{parl['odds']:.2f}"):
                            for leg in parl['combo']:
                                st.markdown(f"- **{leg['match']}**: {leg['desc']} *(Cuota: {leg['odds']:.2f})*")
                            st.caption(f"Beneficio Esperado (EV): {parl['ev']:.3f} (Mientras mayor sea el número, mejor valor tiene contra la casa de apuestas).")

# --- PESTAÑA 6: REGISTRO HISTÓRICO ---
with tabs[5]:
    st.header("📊 Registro Histórico")
    df_preds = tracker.load_predictions()
    if df_preds.empty:
        st.info("No hay predicciones registradas aún.")
    else:
        st.dataframe(df_preds, use_container_width=True)

# --- PESTAÑA 7: INFO DEL MODELO ---
with tabs[6]:
    st.header("📈 Desempeño del Modelo")
    c1, c2 = st.columns(2)
    c1.metric("MAE Local (Error promedio)", f"{metrics['mae_home']:.2f} pts")
    c2.metric("MAE Visitante (Error promedio)", f"{metrics['mae_away']:.2f} pts")

# CAMBIO 2: Lógica de la pestaña de Diagnóstico agregada al final
# --- PESTAÑA 8: DIAGNÓSTICO ---
with tabs[7]:
    st.header("🐛 Diagnóstico de Características y Predicciones Base")
    st.write(f"Monitor de variables para: Temporada {selected_season} - Semana {selected_week}")
    
    if week_games.empty:
        st.info("No hay partidos en esta semana.")
    else:
        diag_data = []
        for _, row in week_games.iterrows():
            g_diag = matchups[matchups['game_id'] == row['game_id']].iloc[0]
            X_diag = g_diag[feat_cols].to_frame().T.fillna(0)
            
            p_h = max(0, m_home.predict(X_diag)[0])
            p_a = max(0, m_away.predict(X_diag)[0])
            
            # Simulación rápida (solo para lectura visual en diagnóstico)
            res_diag = monte_carlo.run_simulation(p_h, p_a, std_home, std_away, n_sims=1000)
            
            diag_data.append({
                "Partido": f"{row['away_team']} @ {row['home_team']}",
                "Pred. Local Base": round(p_h, 2),
                "Pred. Vis Base": round(p_a, 2),
                "Prob. Local": f"{res_diag['prob_home']*100:.1f}%",
                "Prob. Vis": f"{res_diag['prob_away']*100:.1f}%",
                "Pts Temp Local": round(g_diag.get('home_pts_scored_season', 0), 2),
                "Mom. Off Local": round(g_diag.get('home_momentum_off', 0), 2),
                "Pts Temp Vis": round(g_diag.get('away_pts_scored_season', 0), 2),
                "Mom. Off Vis": round(g_diag.get('away_momentum_off', 0), 2)
            })
        
        st.dataframe(pd.DataFrame(diag_data), use_container_width=True)
        
