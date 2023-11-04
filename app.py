"""
__author__ = "Yannis Schmutz"
__copyright__ = "Copyright 2023"
__credits__ = ["Yannis Schmutz"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Yannis Schmutz"
__email__ = "yannis.schmutz@gmail.com"
__status__ = "Beta"
"""

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import date

from data_loading import load_full_reconstruction, load_loo_reconstruction, load_ground_truth, get_station_indices_map
from taylor_helpers import get_taylor_fig
from my_plotting import display_predictions, create_station_line_plot
from config import get_config, DATA_SET_KEY_NAME_MAP, DATA_SET_NAME_KEY_MAP, DATA_SET_DESCRIPTIONS
from data_transformer import extract_stations_from_nc

conf = get_config()

st.set_page_config(layout="wide")


st.write("# Validation Tool Weather Reconstruction 1807")
st.write("This tool lets you select different reconstructions_cnf of the year 1807 "
         "done in the scope of a master thesis by Yannis Schmutz.")


# === Select and read Reconstruction Data =============

reconstructions_cnf = conf['data']['reconstructions']
reconstruction_selections = (DATA_SET_KEY_NAME_MAP['plain'],
                             DATA_SET_KEY_NAME_MAP['analog_3p'],
                             DATA_SET_KEY_NAME_MAP['analog_3p_WT'],
                             )
col_recon_select1, col_recon_select2 = st.columns(2)
with col_recon_select1:
    chosen_reconstruction = st.radio("Select a reconstruction", options=reconstruction_selections)
    chosen_reconstruction = DATA_SET_NAME_KEY_MAP[chosen_reconstruction]
with col_recon_select2:
    st.write(f"**{DATA_SET_KEY_NAME_MAP['plain']}:** {DATA_SET_DESCRIPTIONS['plain']}")
    st.write(f"**{DATA_SET_KEY_NAME_MAP['analog_3p']}:** {DATA_SET_DESCRIPTIONS['analog_3p']}")
    st.write(f"**{DATA_SET_KEY_NAME_MAP['analog_3p_WT']}:** {DATA_SET_DESCRIPTIONS['analog_3p_WT']}")

full_reconstruction = load_full_reconstruction(chosen_reconstruction, reconstructions_cnf)
loo_reconstruction = load_loo_reconstruction(chosen_reconstruction, reconstructions_cnf)
ground_truth = load_ground_truth(conf['data'])
# ======================================================================================================================

col_quantitative, col_qualitative = st.columns(2)
with col_quantitative:
    st.write("## Quantitative Validation")
    quant_selection_col1, quant_selection_col2, quant_selection_col3 = st.columns(3)
    with quant_selection_col1:
        quant_start_date = st.date_input("Choose a start date for validation",
                                         value=date(1807, 1, 1),
                                         min_value=date(1807, 1, 1),
                                         max_value=date(1807, 12, 31),
                                         format="YYYY-MM-DD")
    with quant_selection_col2:
        quant_end_date = st.date_input("Choose a end date for validation",
                                       value=date(1807, 12, 31),
                                       min_value=quant_start_date,
                                       max_value=date(1807, 12, 31),
                                       format="YYYY-MM-DD")
    with quant_selection_col3:
        quant_variable = st.radio("Select variable(s)", options=["both", "ta", "slp"])

    fig_taylor = get_taylor_fig(ground_truth, loo_reconstruction, quant_variable, quant_start_date, quant_end_date)
    st.pyplot(fig_taylor)


with col_qualitative:
    st.write("## Qualitative Validation")
    qual_start_date = st.date_input("Choose a start date for prediction visualization",
                                     value=date(1807, 1, 1),
                                     min_value=date(1807, 1, 1),
                                     max_value=date(1807, 12, 27),
                                     format="YYYY-MM-DD")
    qual_fig = display_predictions(full_reconstruction, qual_start_date)
    st.pyplot(qual_fig)

    st.write("### Single Station Inspection")

    station_indx_map = get_station_indices_map("1807")
    gt_stations = extract_stations_from_nc(ground_truth, station_indx_map)
    pred_stations = extract_stations_from_nc(loo_reconstruction, station_indx_map)

    chosen_station = st.selectbox("Chose a station", gt_stations.keys())

    fig_station = create_station_line_plot(pred_stations[chosen_station], gt_stations[chosen_station], chosen_station)
    st.pyplot(fig_station)
