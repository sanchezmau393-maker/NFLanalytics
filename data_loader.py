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
    try:
        # Intenta cargar la lista original (ej. [2026, 2025])
        df = nfl.import_weekly_data(years_list)
        return df
    except Exception as e:
        # Si da error 404 y hay más de un año en la lista, quitamos el año futuro/actual sin datos
        if "404" in str(e) and len(years_list) > 1:
            years_list.pop(0) # Elimina el año más reciente (ej. 2026)
            try:
                df = nfl.import_weekly_data(years_list)
                return df
            except Exception as e2:
                st.warning(f"No se pudieron cargar datos de jugadores de respaldo: {e2}")
                return pd.DataFrame()
        else:
            st.warning(f"No se pudieron cargar datos de jugadores: {e}")
            return pd.DataFrame()
