import streamlit as st
import nfl_data_py as nfl
import pandas as pd

@st.cache_data(show_spinner=False)
def load_historical_schedules(start_year=2015, end_year=2025):
    try:
        years = list(range(start_year, end_year + 1))
        df = nfl.import_schedules(years)
        return df
    except Exception as e:
        st.error(f"Error crítico cargando historial de partidos: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_weekly_data(year):
    try:
        df = nfl.import_weekly_data([year])
        return df
    except Exception as e:
        st.warning(f"No se pudieron cargar datos semanales para {year}: {e}")
        return pd.DataFrame()
