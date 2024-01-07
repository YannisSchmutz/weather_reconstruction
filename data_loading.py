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


def get_station_indices_map():
    """
    Returns dataframe referencing the station locations.
    :return:
    """
    df = pd.read_csv("data/metadata/station_1807_matrix_indices.csv")
    return df


def date_to_id(date):
    dates = pd.date_range('1807-01-01', freq='D', periods=365).values
    dates = list(map(lambda d: str(d).split('T')[0], dates))
    return dates.index(date)
