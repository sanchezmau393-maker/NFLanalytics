import streamlit as st
import pandas as pd
import numpy as np
import data_loader
import features
import model
import monte_carlo
import tracker

st.set_page_config(page_title="NFL Analytics Pro", page_icon="🏈", layout="wide")

# Inicialización de estado
if 'selected_game_id' not in st.session_state:
    st.session_state.selected_game_id = None

def select_game(game_id):
    st.session_state.selected_game_id = game_id

st.title("🏈 NFL Analytics Pro")
st.markdown("Análisis avanzado y predicción de la NFL mediante Machine Learning y simulaciones Monte Carlo.")

# --- 1. CARGA DE DATOS (Con Spinner, evita pantalla negra) ---
with st.spinner("Descargando/Verificando históricos de la NFL (Caché activo)..."):
    schedules = data_loader.load_historical_schedules(2015, 2025)
    if schedules.empty:
        st.error("Error crítico: No se pudieron cargar los datos. Revisa la conexión o compatibilidad de nfl_data_py.")
        st.stop()

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (Evita Data Leakage) ---
with st.spinner("Procesando momentum y estadísticas previas a los encuentros..."):
    team_games = features.build_features(schedules)
    matchups = features.prepare_matchup_data(schedules, team_games)

# --- 3. ENTRENAMIENTO DE MODELO ---
with st.spinner("Entrenando modelos basados en árboles (HistGradientBoosting)..."):
    m_home, m_away, feat_cols, std_home, std_away, metrics = model.train_models(matchups)

# --- INTERFAZ PRINCIPAL ---
st.sidebar.header("⚙️ Configuración")
seasons_avail = sorted(schedules['season'].dropna().unique(), reverse=True)
selected_season = st.sidebar.selectbox("Temporada:", seasons_avail, index=0)

season_data = matchups[matchups['season'] == selected_season]
weeks_avail = sorted(season_data['week'].dropna().unique())
# Determinar semana actual: la primera semana con partidos sin resultado
unplayed = season_data[season_data['home_score'].isna()]
current_week = int(unplayed['week'].min()) if not unplayed.empty else (weeks_avail[-1] if weeks_avail else 1)

selected_week = st.sidebar.selectbox("Semana:", weeks_avail, index=weeks_avail.index(current_week) if current_week in weeks_avail else 0)

tabs = st.tabs([
    "📅 Cartelera", 
    "🔮 Predicción de Partido", 
    "🔗 Combinadas (Parlays)",
    "📊 Registro de Predicciones", 
    "📈 Info del Modelo",
    "🏃 Player Props"
])

week_games = season_data[season_data['week'] == selected_week]

# --- PESTAÑA 1: CARTELERA ---
with tabs[0]:
    st.header(f"Enfrentamientos - Semana {int(selected_week)}")
    if week_games.empty:
        st.warning("No hay partidos programados o datos disponibles para esta semana.")
    else:
        cols = st.columns(3)
        for idx, row in week_games.iterrows():
            col = cols[idx % 3]
            with col:
                st.container(border=True)
                st.markdown(f"### {row['away_team']} @ {row['home_team']}")
                home_rec = row.get('home_record', 'N/A')
                st.caption("Haz clic para abrir el análisis profundo.")
                if st.button(f"Analizar {row['away_team']} vs {row['home_team']}", key=f"btn_{row['game_id']}", use_container_width=True):
                    select_game(row['game_id'])
                    st.success("Partido seleccionado. Ve a la pestaña '🔮 Predicción de Partido'.")

# --- PESTAÑA 2: PREDICCIÓN ---
with tabs[1]:
    if not st.session_state.selected_game_id:
        st.info("👈 Selecciona un partido desde la pestaña '📅 Cartelera'.")
    else:
        g = matchups[matchups['game_id'] == st.session_state.selected_game_id]
        if g.empty:
            st.error("Partido no encontrado en los registros.")
        else:
            g = g.iloc[0]
            st.header(f"Análisis: {g['away_team']} @ {g['home_team']}")
            
            col_aj1, col_aj2 = st.columns(2)
            with col_aj1:
                st.subheader(f"🛠 Ajustes {g['home_team']}")
                adj_home = st.number_input(f"Modificador Pts {g['home_team']} (Ej: -3 por lesión QB)", value=0.0, step=0.5)
            with col_aj2:
                st.subheader(f"🛠 Ajustes {g['away_team']}")
                adj_away = st.number_input(f"Modificador Pts {g['away_team']} (Ej: -3 por lesión QB)", value=0.0, step=0.5)
            
            # Predicción Base
            X_match = g[feat_cols].to_frame().T.fillna(0)
            pred_h_base = max(0, m_home.predict(X_match)[0])
            pred_a_base = max(0, m_away.predict(X_match)[0])
            
            pred_h = max(0, pred_h_base + adj_home)
            pred_a = max(0, pred_a_base + adj_away)
            
            st.markdown("---")
            st.subheader("🎲 Simulaciones Monte Carlo")
            col_L1, col_L2, col_L3 = st.columns(3)
            
            auto_ou = float(g['total_line']) if pd.notna(g['total_line']) else 45.5
            auto_spread = float(g['spread_line']) if pd.notna(g['spread_line']) else 0.0
            
            ou_line = col_L1.number_input("Línea Over/Under", value=auto_ou)
            spread_line = col_L2.number_input("Spread (Puntos a favor/contra del Local)", value=auto_spread)
            n_sims = col_L3.selectbox("Número de Simulaciones", [1000, 10000, 50000, 100000], index=1)
            
            if st.button("🚀 Ejecutar Predicción y Simulación", type="primary"):
                res = monte_carlo.run_simulation(pred_h, pred_a, std_home, std_away, n_sims=n_sims)
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Victoria {g['home_team']}", f"{res['prob_home']*100:.1f}%")
                c2.metric("Empate", f"{res['prob_tie']*100:.1f}%")
                c3.metric(f"Victoria {g['away_team']}", f"{res['prob_away']*100:.1f}%")
                
                st.write(f"### 🎯 Marcador Esperado: {g['home_team']} {pred_h:.1f} - {g['away_team']} {pred_a:.1f}")
                
                # Análisis O/U y Spread
                prob_over = np.mean(res['total'] > ou_line)
                prob_under = np.mean(res['total'] < ou_line)
                prob_cover = np.mean(res['diff'] > spread_line) # ej: spread_line = -3.5. diff (local - vis) debe ser > -3.5
                
                st.markdown(f"**Over {ou_line}:** {prob_over*100:.1f}%  |  **Under {ou_line}:** {prob_under*100:.1f}%")
                st.markdown(f"**Probabilidad {g['home_team']} cubre Spread ({spread_line}):** {prob_cover*100:.1f}%")
                
                tracker.save_prediction(
                    g['game_id'], g['season'], g['week'], g['home_team'], g['away_team'],
                    pred_h, pred_a, res['prob_home'], res['prob_away'], ou_line, spread_line
                )
                st.success("✅ Predicción ejecutada y registrada en el historial.")

# --- PESTAÑA 3: COMBINADAS ---
with tabs[2]:
    st.header("🔗 Calculadora de Parlays (Combinadas)")
    st.warning("⚠️ Importante: El cálculo asume independencia entre selecciones. Las selecciones del mismo partido (Over + Ganador) están fuertemente correlacionadas. Usar solo como estimación.")
    
    df_preds = tracker.load_predictions()
    if not df_preds.empty:
        selections = st.multiselect("Selecciona partidos guardados en el registro:", df_preds['game_id'].tolist())
        if selections:
            prob_total = 1.0
            st.markdown("### Selecciones:")
            for s in selections:
                row = df_preds[df_preds['game_id'] == s].iloc[0]
                max_p = max(row['prob_home'], row['prob_away'])
                pick_team = row['home_team'] if row['prob_home'] > row['prob_away'] else row['away_team']
                prob_total *= max_p
                st.write(f"- **{s}**: Gana {pick_team} (Prob: {max_p*100:.1f}%)")
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Probabilidad Combinada Estimada", f"{prob_total*100:.2f}%")
            c2.metric("Cuota Justa Implícita (Decimal)", f"{1/prob_total:.2f}" if prob_total > 0 else "N/A")
    else:
        st.info("Genera y guarda predicciones en la pestaña 'Predicción' para usar este módulo.")

# --- PESTAÑA 4: REGISTRO ---
with tabs[3]:
    st.header("📊 Registro Histórico de Predicciones")
    df_preds = tracker.load_predictions()
    if df_preds.empty:
        st.info("No hay predicciones en el sistema.")
    else:
        st.dataframe(df_preds, use_container_width=True)

# --- PESTAÑA 5: INFO DEL MODELO ---
with tabs[4]:
    st.header("📈 Validación Temporal y Desempeño")
    st.write("El modelo utiliza `HistGradientBoostingRegressor`, una técnica de árboles robusta para datos tabulares.")
    st.write("Variables integradas:")
    st.write("- Promedios móviles de temporada (para evitar Data Leakage).")
    st.write("- Desempeño de últimos 3 y 5 partidos (Momentum).")
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Error Absoluto Medio (Local)", f"{metrics['mae_home']:.2f} pts")
    c2.metric("Error Absoluto Medio (Visitante)", f"{metrics['mae_away']:.2f} pts")
    st.caption("El Error Absoluto Medio (MAE) indica la variación promedio entre los puntos predichos y los puntos reales sobre los datos históricos conocidos.")

# --- PESTAÑA 6: PLAYER PROPS ---
with tabs[5]:
    st.header("🏃 Player Props")
    st.info("El módulo de Player Props requiere la carga de métricas detalladas jugada-por-jugada. La arquitectura permite escalabilidad futura, pero temporalmente se mantiene separado para garantizar una carga veloz e interfaz sin 'pantallas negras'.")
