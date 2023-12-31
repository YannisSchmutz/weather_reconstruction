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

from data_loading import load_full_reconstruction, load_loo_reconstruction, load_ground_truth, \
    get_station_indices_map, date_to_id
from taylor_helpers import get_taylor_fig
from my_plotting import display_predictions, create_station_line_plot
from config import get_config, DATA_SET_KEY_NAME_MAP, DATA_SET_NAME_KEY_MAP, DATA_SET_DESCRIPTIONS
from data_transformer import extract_stations_from_nc
from config import CITIES


conf = get_config()

st.set_page_config(layout="wide", page_title="Weather Reconstruction 1807", page_icon="🌦️")

st.write("# Validation Tool Weather Reconstruction 1807")
st.write("This tool lets you select different weather reconstructions of the year 1807 "
         "done in the scope of a master thesis by Yannis Schmutz.")


# === Select and read Reconstruction Data =============

reconstructions_cnf = conf['data']['reconstructions']
reconstruction_selections = (DATA_SET_KEY_NAME_MAP['plain'],
                             DATA_SET_KEY_NAME_MAP['arm'],
                             DATA_SET_KEY_NAME_MAP['arm_wt'],
                             )
col_recon_select1, col_recon_select2 = st.columns([1, 3])
with col_recon_select1:
    chosen_reconstruction = st.radio("#### Select a reconstruction", options=reconstruction_selections,
                                     help=f"**{DATA_SET_KEY_NAME_MAP['plain']}:** {DATA_SET_DESCRIPTIONS['plain']}." +
                                     f"**{DATA_SET_KEY_NAME_MAP['arm']}:** {DATA_SET_DESCRIPTIONS['arm']}." +
                                     f"**{DATA_SET_KEY_NAME_MAP['arm_wt']}:** {DATA_SET_DESCRIPTIONS['arm_wt']}")
    chosen_reconstruction = DATA_SET_NAME_KEY_MAP[chosen_reconstruction]

full_reconstruction = load_full_reconstruction(chosen_reconstruction, reconstructions_cnf)
loo_reconstruction = load_loo_reconstruction(chosen_reconstruction, reconstructions_cnf)
ground_truth = load_ground_truth(conf['data'])
# ======================================================================================================================

tab1, tab2, tab3 = st.tabs(["Quantitative Validation", "Qualitative Validation", "Single Station Inspection"])

with tab1:
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

    # Ugly workaround for controling figure size
    col_taylor, _, _ = st.columns([2, 1, 1])
    with col_taylor:
        fig_taylor = get_taylor_fig(ground_truth, loo_reconstruction, quant_variable, quant_start_date, quant_end_date)
        st.pyplot(fig_taylor)
# ----------------------------------------------------------------------------------------------------------------------

with tab2:
    st.write("## Qualitative Validation")
    st.write("Visualizes the reconstructed temperature (top) in °C and pressure (bottom) in hPa")
    qual_start_date = st.date_input("Choose a start date for prediction visualization",
                                    value=date(1807, 1, 1),
                                    min_value=date(1807, 1, 1),
                                    max_value=date(1807, 12, 27),
                                    format="YYYY-MM-DD")
    date_str = qual_start_date.strftime("%Y-%m-%d")
    date_id = date_to_id(date_str)

    show_contours = st.toggle('Show reconstructions as contour map', value=True)

    qual_fig = display_predictions(full_reconstruction, date_id, date_str, show_contours=show_contours)
    st.pyplot(qual_fig)
# ----------------------------------------------------------------------------------------------------------------------


with tab3:
    st.write("## Single Station Inspection")
    st.write("The line chart shows the historic observation versus the LOO-reconstructed time series.")

    station_indx_map = get_station_indices_map()
    gt_stations = extract_stations_from_nc(ground_truth, station_indx_map)
    pred_stations = extract_stations_from_nc(loo_reconstruction, station_indx_map)

    chosen_station = st.selectbox("Chose a station", gt_stations.keys())

    fig_station = create_station_line_plot(pred_stations[chosen_station], gt_stations[chosen_station], chosen_station)
    st.pyplot(fig_station)
# ----------------------------------------------------------------------------------------------------------------------



