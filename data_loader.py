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

@st.cache_data(show_spinner=False)
def load_weekly_data(years_list):
    df_list = []
    for year in years_list:
        try:
            df = nfl.import_weekly_data([year])
            if not df.empty:
                df_list.append(df)
        except Exception:
            # Si el archivo de la temporada (ej. 2026) no existe (Error 404), 
            # lo ignoramos silenciosamente y probamos con el año anterior.
            continue 
            
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    
    return pd.DataFrame()
