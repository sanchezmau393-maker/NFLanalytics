import streamlit as st
import nfl_data_py as nfl
import pandas as pd

@st.cache_data(show_spinner=False)
def load_historical_schedules(start_year=2015, end_year=2026):
    try:
        return nfl.import_schedules(list(range(start_year, end_year + 1)))
    except Exception:
        return pd.DataFrame()

# CACHÉ PERMANENTE: Los años pasados no cambian, se cargan una vez al iniciar la app.
@st.cache_data(show_spinner=False)
def _load_historical_player_stats(years_list):
    try:
        return nfl.import_weekly_data(years_list, downcast=True)
    except Exception:
        return pd.DataFrame()

# CACHÉ TEMPORAL (30 min): Solo recargamos el año actual para ver resultados recientes.
@st.cache_data(show_spinner=False, ttl=1800)
def _load_current_player_stats(year):
    try:
        return nfl.import_weekly_data([year], downcast=True)
    except Exception:
        return pd.DataFrame()

def load_all_player_stats(years_list):
    if not years_list: return pd.DataFrame()
    curr_year = max(years_list)
    hist_years = [y for y in years_list if y != curr_year]
    
    df_hist = _load_historical_player_stats(hist_years) if hist_years else pd.DataFrame()
    df_curr = _load_current_player_stats(curr_year)
    
    frames = [df for df in [df_hist, df_curr] if not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ROSTERS ACTUALES: Descargamos el roster oficial para evitar jugadores en equipos antiguos.
@st.cache_data(show_spinner=False, ttl=86400)
def load_current_rosters(year):
    try:
        df = nfl.import_rosters([year])
        if 'player_name' in df.columns:
            df['player_display_name'] = df['player_name']
        # Filtramos para tener la dupla Jugador-Equipo actual
        return df[['player_display_name', 'team', 'position', 'status']].dropna(subset=['team'])
    except Exception:
        return pd.DataFrame()
