import numpy as np
import xarray as xr
import pandas as pd

import streamlit as st


@st.cache_data
def load_full_reconstruction(rec_conf_id, rec_conf):
    full_rec = np.load(rec_conf['base_path'] + rec_conf[rec_conf_id]['full'])
    return full_rec


# @st.cache_data
def load_loo_reconstruction(rec_conf_id, rec_conf):
    loo_rec = xr.open_dataset(rec_conf['base_path'] + rec_conf[rec_conf_id]['loo'])
    return loo_rec


# @st.cache_data
def load_ground_truth(data_conf):
    return xr.open_dataset(data_conf['base_path'] + data_conf['ground_truth'])


# TODO: from config
STATION_INDICES = "data/metadata/station_{}_matrix_indices.csv"
def get_station_indices_map(year):
    """
    Returns dataframe referencing the station locations depending on <missing_like>
    :param year: Str of target inference year. E.g. "1807"
    :return:
    """
    df = pd.read_csv(STATION_INDICES.format(year))
    return df
