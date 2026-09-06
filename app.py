import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
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

st.title("🏈 NFL Analytics Pro")
st.markdown("Análisis avanzado, predicción objetiva de bajas y Player Props mediante Machine Learning y Monte Carlo.")

# --- 1. CARGA DE DATOS ---
with st.spinner("Descargando calendarios e históricos de la NFL..."):
    schedules = data_loader.load_historical_schedules(2015, 2026)
    if schedules.empty:
        st.error("Error al obtener los calendarios. Por favor recarga la página.")
        st.stop()

# --- 2. INGENIERÍA DE CARACTERÍSTICAS Y ENTRENAMIENTO ---
with st.spinner("Procesando momentum y modelos predictivos..."):
    team_games = features.build_features(schedules)
    matchups = features.prepare_matchup_data(schedules, team_games)
    m_home, m_away, feat_cols, std_home, std_away, metrics = model.train_models(matchups)

# Cargar datos de jugadores para lesiones y props (últimas temporadas)
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

# Detectar semana predeterminada
unplayed = season_data[season_data['home_score'].isna()]
default_week = int(unplayed['week'].min()) if not unplayed.empty else (weeks_avail[0] if weeks_avail else 1)

selected_week = st.sidebar.selectbox("Semana:", weeks_avail, index=weeks_avail.index(default_week) if default_week in weeks_avail else 0)

tabs = st.tabs([
    "📅 Cartelera", 
    "🔮 Predicción", 
    "🎯 Eficiencia",
    "🏃 Player Props",
    "🔗 Combinadas",
    "📊 Registro", 
    "📈 Info Modelo"
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

# --- PESTAÑA 2: PREDICCIÓN CON LESIONES AUTOMÁTICAS ---
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
            
            # Módulo de lesiones por Roster
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
    st.write("Esta pestaña evalúa automáticamente las predicciones del modelo contra los resultados reales de los partidos que ya concluyeron en la temporada seleccionada.")
    
    played_games = season_data[season_data['home_score'].notna()].copy()
    
    if played_games.empty:
        st.info(f"Aún no hay resultados finales para evaluar en la temporada {selected_season}.")
    else:
        # Extraer características y predecir
        X_hist = played_games[feat_cols].fillna(0)
        played_games['pred_home'] = m_home.predict(X_hist)
        played_games['pred_away'] = m_away.predict(X_hist)
        
        # Determinar ganadores reales y predichos
        played_games['real_winner'] = np.where(played_games['home_score'] > played_games['away_score'], played_games['home_team'], played_games['away_team'])
        played_games['pred_winner'] = np.where(played_games['pred_home'] > played_games['pred_away'], played_games['home_team'], played_games['away_team'])
        
        # Evaluar aciertos
        played_games['is_correct'] = played_games['real_winner'] == played_games['pred_winner']
        
        # Excluir empates de la métrica (son muy raros y ensucian la eficacia de ganador)
        empates_reales = played_games['home_score'] == played_games['away_score']
        valid_games = played_games[~empates_reales]
        
        total_valid = len(valid_games)
        correct_preds = valid_games['is_correct'].sum()
        acc_percentage = (correct_preds / total_valid * 100) if total_valid > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos Evaluados", total_valid)
        c2.metric("Predicciones Acertadas", correct_preds)
        c3.metric("Eficacia General (%)", f"{acc_percentage:.1f}%")
        
        st.subheader("Evolución de Eficacia por Semana")
        # Agrupar por semana para el gráfico
        weekly_acc = valid_games.groupby('week')['is_correct'].mean() * 100
        weekly_acc.name = "% de Acierto"
        st.line_chart(weekly_acc)
        
        st.subheader("Desglose de Partidos")
        display_df = played_games[['week', 'away_team', 'home_team', 'away_score', 'home_score', 'pred_winner', 'is_correct']].copy()
        display_df.rename(columns={
            'week': 'Semana',
            'away_team': 'Visitante',
            'home_team': 'Local',
            'away_score': 'Pts Vis',
            'home_score': 'Pts Loc',
            'pred_winner': 'Ganador Predicho',
            'is_correct': '¿Acertó?'
        }, inplace=True)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- PESTAÑA 4: PLAYER PROPS FUNCIONAL ---
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
            if pos == 'QB':
                metrics_list = [('passing_yards', 'Yardas por Pase'), ('attempts', 'Intentos de Pase')]
            elif pos in ['WR', 'TE']:
                metrics_list = [('receiving_yards', 'Yardas por Recepción'), ('receptions', 'Recepciones')]
            elif pos == 'RB':
                metrics_list = [('rushing_yards', 'Yardas Terrestres'), ('carries', 'Acarreos')]
            else:
                metrics_list = [('receiving_yards', 'Yardas Totales'), ('receptions', 'Recepciones')]

            probs = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
            labels = ["90% (Muy Seguro)", "80%", "70%", "60%", "50% (Promedio)", "40%", "30%", "20%", "10% (Arriesgado)"]
            
            df_props = pd.DataFrame({"Probabilidad (OVER)": labels})
            
            for col_stat, stat_name in metrics_list:
                stat_values = p_data[col_stat].dropna()
                if len(stat_values) >= 2:
                    mu, sigma = stat_values.mean(), stat_values.std()
                    if sigma == 0 or np.isnan(sigma): sigma = 0.1
                    lines = [max(0, np.round(norm.ppf(1 - p, loc=mu, scale=sigma), 1)) for p in probs]
                    df_props[f"Línea de {stat_name}"] = lines
                    st.caption(f"**{stat_name}:** Promedio {mu:.1f} | Desviación {sigma:.1f}")
            
            st.dataframe(df_props, use_container_width=True, hide_index=True)

# --- PESTAÑA 5: COMBINADAS ---
with tabs[4]:
    st.header("🔗 Calculadora de Parlays")
    df_preds = tracker.load_predictions()
    if not df_preds.empty:
        selections = st.multiselect("Selecciona partidos guardados:", df_preds['game_id'].tolist())
        if selections:
            prob_total = 1.0
            for s in selections:
                row = df_preds[df_preds['game_id'] == s].iloc[0]
                max_p = max(row['prob_home'], row['prob_away'])
                pick_team = row['home_team'] if row['prob_home'] > row['prob_away'] else row['away_team']
                prob_total *= max_p
                st.write(f"- **{s}**: Gana {pick_team} ({max_p*100:.1f}%)")
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Probabilidad Combinada", f"{prob_total*100:.2f}%")
            c2.metric("Cuota Implícita (Decimal)", f"{1/prob_total:.2f}" if prob_total > 0 else "N/A")
    else:
        st.info("Realiza y guarda predicciones para armar combinadas.")

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
