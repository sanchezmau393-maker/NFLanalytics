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

@st.cache_data(show_spinner=False, ttl=3600)
def load_all_player_stats(years_list):
    df_list = []
    for year in years_list:
        try:
            df = nfl.import_weekly_data([year], downcast=True)
            if df is not None and not df.empty:
                df_list.append(df)
        except Exception:
            continue 
            
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    
    return pd.DataFrame()
