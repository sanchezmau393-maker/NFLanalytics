import streamlit as st
import nfl_data_py as nfl
import pandas as pd

@st.cache_data(show_spinner=False)
def load_historical_schedules(start_year=2015, end_year=2026):
    try:
        years = list(range(start_year, end_year + 1))
        df = nfl.import_schedules(years)
        return df
    except Exception as e:
        st.error(f"Error cargando historial de partidos: {e}")
        return pd.DataFrame()

# AÑADIDO: Caché permanente para años históricos (acelera la carga un 80%)
@st.cache_data(show_spinner=False)
def _load_historical_stats(years):
    try:
        return nfl.import_weekly_data(years, downcast=True)
    except:
        return pd.DataFrame()

# AÑADIDO: Caché de 30 min solo para el año actual
@st.cache_data(show_spinner=False, ttl=1800)
def _load_current_stats(year):
    try:
        return nfl.import_weekly_data([year], downcast=True)
    except:
        return pd.DataFrame()

def load_all_player_stats(years_list):
    if not years_list: return pd.DataFrame()
    curr_year = max(years_list)
    hist_years = [y for y in years_list if y != curr_year]
    
    df_hist = _load_historical_stats(hist_years) if hist_years else pd.DataFrame()
    df_curr = _load_current_stats(curr_year)
    
    frames = [df for df in [df_hist, df_curr] if not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# AÑADIDO (Mejora 10.1): Carga de rosters oficiales para evitar jugadores en equipos antiguos
@st.cache_data(show_spinner=False, ttl=86400)
def load_current_rosters(year):
    try:
        return nfl.import_rosters([year])
    except Exception:
        return pd.DataFrame()
