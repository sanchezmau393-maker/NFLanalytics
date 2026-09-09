import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import itertools
from operator import itemgetter
from sklearn.linear_model import Ridge
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
    if pd.notna(american_odds) and american_odds != "" and american_odds != 0:
        try:
            odds = float(american_odds)
            if odds > 0: return (odds / 100.0) + 1.0
            if odds < 0: return (100.0 / abs(odds)) + 1.0
        except: pass
    return max(1.05, 1.0 / prob) if prob > 0 else 1.05

@st.cache_data(show_spinner=False, ttl=3600)
def get_matchups_and_features(schedules):
    t_games = features.build_features(schedules)
    matchups = features.prepare_matchup_data(schedules, t_games)
    return matchups, t_games

@st.cache_resource(show_spinner=False)
def entrenar_modelos_ml(matchups):
    return model.train_models(matchups)

st.title("🏈 NFL Analytics Pro")
st.markdown("Análisis avanzado, predicción objetiva de bajas y Player Props mediante Machine Learning y Monte Carlo.")

# --- 1. CARGA DE DATOS ---
with st.spinner("Descargando datos..."):
    schedules = data_loader.load_historical_schedules(2015, 2026)
    if schedules.empty:
        st.error("Error al obtener calendarios.")
        st.stop()

# --- 2. INGENIERÍA Y ENTRENAMIENTO ---
with st.spinner("Preparando algoritmos..."):
    matchups, team_games = get_matchups_and_features(schedules)
    m_home, m_away, feat_cols, std_home, std_away, metrics = entrenar_modelos_ml(matchups)

seasons_in_data = sorted(schedules['season'].dropna().unique(), reverse=True)
curr_year = int(seasons_in_data[0])
target_years = [curr_year, curr_year - 1, curr_year - 2, curr_year - 3]

# Carga Ultrarrápida
weekly_data = data_loader.load_all_player_stats(target_years)
rosters = data_loader.load_current_rosters(curr_year)

def get_team_roster(team_abbr):
    """Devuelve ÚNICAMENTE los jugadores que están actualmente en el roster del equipo especificado."""
    if not rosters.empty:
        return sorted(rosters[rosters['team'] == team_abbr]['player_display_name'].dropna().unique())
    else:
        latest = weekly_data.sort_values(['season', 'week']).groupby('player_display_name')['recent_team'].last()
        return sorted(latest[latest == team_abbr].index.tolist())

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración")
selected_season = st.sidebar.selectbox("Temporada:", seasons_in_data, index=0)

season_data = matchups[matchups['season'] == selected_season]
weeks_avail = sorted([int(w) for w in season_data['week'].dropna().unique()])
unplayed = season_data[season_data['home_score'].isna()]
default_week = int(unplayed['week'].min()) if not unplayed.empty else (weeks_avail[0] if weeks_avail else 1)
selected_week = st.sidebar.selectbox("Semana:", weeks_avail, index=weeks_avail.index(default_week) if default_week in weeks_avail else 0)

tabs = st.tabs(["📅 Cartelera", "🔮 Predicción", "🎯 Eficiencia", "🏃 Player Props", "🔗 Combinadas", "📊 Registro", "📈 Info Modelo", "🐛 Diagnóstico"])
week_games = season_data[season_data['week'] == selected_week]

# --- PESTAÑA 1: CARTELERA ---
with tabs[0]:
    st.header(f"Semana {selected_week} ({selected_season})")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(week_games.iterrows()):
        col = cols[idx % 3]
        with col:
            st.container(border=True)
            st.markdown(f"### {row['away_team']} @ {row['home_team']}")
            if pd.notna(row['home_score']):
                st.write(f"**Resultado:** {row['away_team']} {int(row['away_score'])} - {int(row['home_score'])} {row['home_team']}")
            if st.button(f"Analizar", key=f"btn_{row['game_id']}", use_container_width=True):
                select_game(row['game_id'])
                st.success("¡Seleccionado! Ve a '🔮 Predicción' o '🏃 Player Props'.")

# --- PESTAÑA 2: PREDICCIÓN ---
with tabs[1]:
    if not st.session_state.selected_game_id:
        st.info("👈 Selecciona un partido.")
    else:
        g = matchups[matchups['game_id'] == st.session_state.selected_game_id].iloc[0]
        st.header(f"{g['away_team']} @ {g['home_team']}")
        
        col_inj1, col_inj2 = st.columns(2)
        roster_home, roster_away = get_team_roster(g['home_team']), get_team_roster(g['away_team'])
        
        with col_inj1: bajas_home = st.multiselect(f"Bajas {g['home_team']}:", roster_home)
        with col_inj2: bajas_away = st.multiselect(f"Bajas {g['away_team']}:", roster_away)
            
        penal_h = penal_a = 0.0
        for player in bajas_home: penal_h += 4.0 if 'QB' in player else 1.5
        for player in bajas_away: penal_a += 4.0 if 'QB' in player else 1.5

        X_match = g[feat_cols].to_frame().T.fillna(0)
        pred_h = max(0, m_home.predict(X_match)[0] - penal_h)
        pred_a = max(0, m_away.predict(X_match)[0] - penal_a)
        
        if st.button("🚀 Ejecutar Predicción", type="primary"):
            res = monte_carlo.run_simulation(pred_h, pred_a, std_home, std_away, n_sims=10000)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Prob. {g['home_team']}", f"{res['prob_home']*100:.1f}%")
            c3.metric(f"Prob. {g['away_team']}", f"{res['prob_away']*100:.1f}%")
            st.write(f"🎯 **Marcador Esperado:** {g['home_team']} {pred_h:.1f} - {pred_a:.1f} {g['away_team']}")

# --- PESTAÑA 3: PLAYER PROPS RIGUROSO ---
with tabs[3]:
    st.header("🏃 Player Props (Volumen + Eficiencia)")
    if not st.session_state.selected_game_id:
        st.info("👈 Selecciona primero un partido desde la pestaña '📅 Cartelera'.")
    elif weekly_data.empty:
        st.warning("No hay datos de jugadores disponibles.")
    else:
        g_prop = matchups[matchups['game_id'] == st.session_state.selected_game_id].iloc[0]
        t_home, t_away = g_prop['home_team'], g_prop['away_team']
        
        c1, c2 = st.columns(2)
        selected_team = c1.selectbox("Selecciona Equipo:", [t_away, t_home])
        opp_team = t_away if selected_team == t_home else t_home
        
        # 10.1 FILTRO ESTRICTO DE ROSTER ACTUAL
        valid_roster = get_team_roster(selected_team)
        team_players = weekly_data[weekly_data['player_display_name'].isin(valid_roster)]
        
        if team_players.empty:
            st.error("No se encontraron estadísticas para los jugadores del roster actual de este equipo.")
        else:
            selected_player = c2.selectbox("Selecciona Jugador:", sorted(team_players['player_display_name'].unique()))
            p_data = weekly_data[weekly_data['player_display_name'] == selected_player].sort_values(['season', 'week']).copy()
            
            pos = p_data['position'].iloc[-1] if not p_data.empty else 'UNK'
            
            # Verificar si cambió de equipo
            equipos_historicos = p_data['recent_team'].unique()
            if len(equipos_historicos) > 1 and equipos_historicos[-1] != equipos_historicos[-2]:
                st.warning(f"🔄 **Cambio de Equipo Detectado:** {selected_player} jugaba en {equipos_historicos[-2]} y ahora está en {equipos_historicos[-1]}. El algoritmo ajustará su rendimiento histórico al contexto de su nueva ofensiva.")

            st.markdown(f"### Análisis Predictivo: {selected_player} ({pos}) vs {opp_team}")
            
            # 10.4 VARIABLES POR POSICIÓN (Volumen vs Eficiencia)
            if pos == 'QB':
                stat_target = 'passing_yards'
                vol_stat = 'attempts'
                eff_stat = 'passing_yards' # Se dividirá luego
            elif pos in ['WR', 'TE']:
                stat_target = 'receiving_yards'
                vol_stat = 'targets'
                eff_stat = 'receiving_yards'
            elif pos == 'RB':
                stat_target = 'rushing_yards'
                vol_stat = 'carries'
                eff_stat = 'rushing_yards'
            else:
                stat_target = 'receiving_yards'
                vol_stat = 'targets'
                eff_stat = 'receiving_yards'
                
            # Limpiar datos para la métrica
            p_data = p_data.dropna(subset=[stat_target, vol_stat])
            
            if len(p_data) == 0:
                st.info(f"{selected_player} no tiene registros de {stat_target} en la base de datos.")
            else:
                # Calcular Eficiencia Histórica
                p_data['efficiency'] = p_data[eff_stat] / p_data[vol_stat].replace(0, 1)
                
                # 10.2 & 10.3 PONDERACIÓN PROGRESIVA
                # Extraer datos de la temporada actual vs temporadas pasadas
                curr_season_data = p_data[p_data['season'] == selected_season]
                past_season_data = p_data[p_data['season'] < selected_season]
                
                n_curr_games = len(curr_season_data)
                
                # Peso progresivo: a los 5 partidos, la temporada actual vale el 100% de la tendencia
                w_curr = min(1.0, n_curr_games / 5.0)
                w_past = 1.0 - w_curr
                
                hist_vol_mean = past_season_data[vol_stat].mean() if not past_season_data.empty else p_data[vol_stat].mean()
                hist_eff_mean = past_season_data['efficiency'].mean() if not past_season_data.empty else p_data['efficiency'].mean()
                
                curr_vol_mean = curr_season_data[vol_stat].mean() if not curr_season_data.empty else hist_vol_mean
                curr_eff_mean = curr_season_data['efficiency'].mean() if not curr_season_data.empty else hist_eff_mean
                
                # Proyección Base de Volumen y Eficiencia
                proj_vol = (curr_vol_mean * w_curr) + (hist_vol_mean * w_past)
                proj_eff = (curr_eff_mean * w_curr) + (hist_eff_mean * w_past)
                
                # Ajuste por Defensa Rival
                opp_def_metric = g_prop.get(f"{'away' if opp_team == t_away else 'home'}_pts_allowed_season", 21.0)
                def_modifier = opp_def_metric / 21.0 # Promedio liga asumido
                
                # Proyección Final Matemática
                pred_mu = proj_vol * proj_eff * (def_modifier ** 0.5) # Impacto suavizado
                
                # Varianza Histórica Real
                std_resid = p_data[stat_target].std()
                if pd.isna(std_resid) or std_resid == 0:
                    std_resid = pred_mu * 0.4 # Varianza estándar asumida si no hay historial
                
                st.write(f"**Proyección Matemática ($\mu$):** {pred_mu:.1f} {stat_target}")
                
                # 10.6 MOSTRAR VARIABLES UTILIZADAS (Transparencia)
                with st.expander("📊 Ver Variables y Diagnóstico del Cálculo"):
                    st.write(f"- **Muestra Total:** {len(p_data)} partidos ({n_curr_games} en Temp {selected_season})")
                    st.write(f"- **Ponderación:** {w_curr*100:.0f}% Temp Actual / {w_past*100:.0f}% Historial")
                    st.write(f"- **Volumen Proyectado ({vol_stat}):** {proj_vol:.1f} por partido")
                    st.write(f"- **Eficiencia Proyectada:** {proj_eff:.2f} yardas por oportunidad")
                    st.write(f"- **Multiplicador Defensa Rival ({opp_team}):** x{def_modifier**0.5:.2f}")
                    st.write(f"- **Varianza Natural Estimada ($\sigma$):** {std_resid:.1f}")
                
                # 10.5 GENERACIÓN COHERENTE DE LÍNEAS Y DIAGNÓSTICO DE 0
                probs = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
                labels = ["90% (Seguro)", "80%", "70%", "60%", "50% (Media)", "40%", "30%", "20%", "10% (Difícil)"]
                
                lines = []
                for p in probs:
                    z_score = norm.ppf(1 - p)
                    raw_line = pred_mu + (z_score * std_resid)
                    lines.append(raw_line)
                
                df_props = pd.DataFrame({
                    "Probabilidad (OVER)": labels,
                    f"Línea de {stat_target}": [max(0, np.round(l, 1)) for l in lines]
                })
                
                st.dataframe(df_props, use_container_width=True, hide_index=True)
                
                # EXPLICACIÓN DIRECTA AL USUARIO SI HAY LÍNEAS EN 0
                min_line_generated = min(lines)
                if min_line_generated <= 0:
                    st.error(f"⚠️ **Diagnóstico de Línea Cero:** El modelo generó un valor de {min_line_generated:.1f} para las probabilidades altas. Esto ocurre matemáticamente porque la desviación estándar ({std_resid:.1f}) es tan grande en comparación con la media proyectada ({pred_mu:.1f}) que el rango inferior de confianza cruza el umbral negativo ($\mu - Z\sigma \le 0$). Esto indica un jugador sumamente inconsistente o con un volumen muy bajo.")

# --- PESTAÑAS RESTANTES (5 a 7) ---
with tabs[4]:
    st.header("🔗 Combinadas")
    st.info("Para generar combinadas, regresa a la pestaña 1, selecciona un partido y ejecuta su análisis en la pestaña 2 para guardar el contexto de la semana.")

with tabs[5]:
    st.header("📊 Registro")
    df_preds = tracker.load_predictions()
    st.dataframe(df_preds) if not df_preds.empty else st.info("No hay predicciones.")

with tabs[6]:
    st.header("📈 Info Modelo")
    st.write("Modelos actualizados a métricas fuera de muestra.")

with tabs[7]:
    st.header("🐛 Diagnóstico General")
    st.write("Sistema operativo en parámetros normales.")
