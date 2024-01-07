
import pandas as pd


# === Leave-one-out helpers ===

def extract_stations_from_nc(data_set, stat_indx_map):
    """
    Returns dict of only station time series of a corresponding xarray data set.
    :param data_set: xarray dataset (could be prediction or gt)
    :param stat_indx_map: Dataframe referencing the station locations.
    :return:
    """
    stations = {}
    for index in stat_indx_map.index:
        station_id = stat_indx_map.loc[index].id
        variable = stat_indx_map.loc[index].variable
        lat = stat_indx_map.loc[index].lat
        lon = stat_indx_map.loc[index].lon

        # just time dim.
        station_val = data_set[variable].sel(dict(lat=lat, lon=lon)).to_numpy()
        stations[f"{station_id}_{variable}"] = station_val
    return stations
