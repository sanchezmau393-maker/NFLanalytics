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

# Cambiamos el nombre a "get_weekly_stats" para forzar a Streamlit a ignorar la caché corrupta
@st.cache_data(show_spinner=False)
def get_weekly_stats(years_list):
    df_list = []
    
    # 1. Intento normal año por año
    for year in years_list:
        try:
            df = nfl.import_weekly_data([year])
            if df is not None and not df.empty:
                df_list.append(df)
        except Exception:
            continue 
            
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    
    # 2. Rescate Forzado: Si la lista original falló por completo, extraemos 2025 y 2024 directamente
    try:
        fallback_df = nfl.import_weekly_data([2025, 2024])
        if fallback_df is not None and not fallback_df.empty:
            return fallback_df
    except Exception:
        pass
        
    return pd.DataFrame()
