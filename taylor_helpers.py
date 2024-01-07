
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import date

from data_loading import get_station_indices_map
from data_transformer import extract_stations_from_nc
from my_plotting import create_normed_taylor_diagram


def get_nan_ids(stations):
    """
    Return for each station the corresponding nan-ids
    :param stations: Extracted stations as dict.
    :return:
    """
    miss_indicies_map = {}
    for station_id, station_vals in stations.items():
        # Remove nan-values in bar.
        missing_indicies = np.argwhere(np.isnan(station_vals))
        if len(missing_indicies):
            missing_indicies = list(np.squeeze(missing_indicies, axis=-1))
        else:
            missing_indicies = []
        miss_indicies_map[station_id] = missing_indicies
    return miss_indicies_map


def handle_saisonality_on_ta(stations, miss_id_map=None, n_doy=365):
    """
    Return stations with seasonal component removed for ta-data.
    :param stations: Extracted stations as dict.
    :param miss_id_map: Nan-ids
    :param n_doy: Number of days in period.
    :return:
    """
    x_ = np.linspace(0, n_doy-1, n_doy)  # doy
    x1 = np.sin((2*np.pi*x_)/n_doy)
    x2 = np.cos((2*np.pi*x_)/n_doy)
    x3 = np.sin((4*np.pi*x_)/n_doy)
    x4 = np.cos((4*np.pi*x_)/n_doy)

    x = pd.DataFrame(data={'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
    x_sm = sm.add_constant(x)

    handled_stations = {}
    for stat_id, stat_vals in stations.items():
        if stat_id.endswith("_ta"):
            tmp_x_sm = x_sm.copy()
            y_obs = np.copy(stat_vals)
            if miss_id_map:
                missing_ids = miss_id_map[stat_id]
                tmp_x_sm = tmp_x_sm.drop(missing_ids)
                y_obs = np.delete(y_obs, missing_ids)

            model = sm.OLS(y_obs, tmp_x_sm).fit()
            c0, c1, c2, c3, c4 = model.params

            saisonal_component = c0 + c1*tmp_x_sm.x1 + c2*tmp_x_sm.x2 + c3*tmp_x_sm.x3 + c4*tmp_x_sm.x4
            desaisonalized = y_obs - saisonal_component
            # Insert nans again:
            if miss_id_map and missing_ids:  # Only gets referenced if miss_id_map is passed...
                nan_vals = pd.Series([np.nan for _ in missing_ids], index=missing_ids)
                desaisonalized = pd.concat([desaisonalized, nan_vals], axis=0, ignore_index=False).sort_index()
            handled_stations[stat_id] = desaisonalized.to_numpy()
        else:
            handled_stations[stat_id] = stat_vals
    return handled_stations


def get_corr(gt, pred):
    """
    Get correlation between ground truth and prediction.
    :param gt: Numpy-gt
    :param pred: Numpy-pred
    :return:
    """
    mu_r = np.mean(gt)
    mu_f = np.mean(pred)
    sigma_r = np.std(gt)
    sigma_f = np.std(pred)

    diff_r = gt - mu_r
    diff_f = pred - mu_f

    ele_wise_mul = diff_r * diff_f
    numerator = np.mean(ele_wise_mul)
    denominator = sigma_r * sigma_f
    corr = numerator / denominator  # TODO: Division through 0 would break this..!
    return corr


def get_normalized_sigma(vector, reference_sigma):
    """
    Normalizes the standard deviation of a given series by another standard deviation.
    :param vector: Series to get the normalized std.dev from
    :param reference_sigma: Reference std.dev
    :return:
    """
    return np.std(vector) / reference_sigma


def get_normalized_rmse(gt, pred):
    """
    Calculates the normalized rmse.
    :param gt: Ground truth
    :param pred: Prediction
    :return:
    """
    mu_r = np.mean(gt)
    mu_f = np.mean(pred)
    sigma_r = np.std(gt)

    diff_r = gt - mu_r
    diff_f = pred - mu_f

    ele_wise_diff = diff_f - diff_r
    ele_wise_square = np.square(ele_wise_diff)
    numerator = np.sqrt(np.mean(ele_wise_square))
    normed_rmse = numerator / sigma_r
    return normed_rmse


def get_loo_taylor_metrics(gt_stations, pred_stations, missing_indicies):
    """
    Returns a dict of dicts of taylor-metrics for each station.
    :param gt_stations: Extracted GT stations
    :param pred_stations: Extracted Pred stations
    :param missing_indicies: Nan-ids
    :return:
    """
    metrics = {}

    for station_id in gt_stations.keys():
        pred = pred_stations[station_id]
        gt = gt_stations[station_id]

        # Remove unobserved value-indicies of GT from pred-series
        pred_clean = np.delete(pred, missing_indicies[station_id])
        gt_clean = np.delete(gt, missing_indicies[station_id])

        corr = get_corr(gt_clean, pred_clean)
        rmse_hat = get_normalized_rmse(gt_clean, pred_clean)

        # sigma_gt_hat = 1
        sigma_pred_hat = get_normalized_sigma(pred_clean, np.std(gt_clean))

        metrics[station_id] = {"corr": corr, "norm_std": sigma_pred_hat, "norm_rmse": rmse_hat}
    return metrics


def _cut_station_dict_to_date_range(station_dict, start_date, end_date):
    whole_range = pd.date_range(date(1807, 1, 1), date(1807, 12, 31), freq="D")
    whole_range = pd.Series(whole_range)
    date_range_ids = whole_range[(whole_range.dt.date >= start_date) & (whole_range.dt.date <= end_date)].index.values
    reduced = {}
    for k, v in station_dict.items():
        reduced[k] = np.array(v)[date_range_ids].tolist()
    return reduced


def _remove_miss_ids_not_in_range(miss_ids, start_date, end_date):
    whole_range = pd.date_range(date(1807, 1, 1), date(1807, 12, 31), freq="D")
    whole_range = pd.Series(whole_range)
    date_range_ids = whole_range[(whole_range.dt.date >= start_date) & (whole_range.dt.date <= end_date)].index.values
    reduced = {}
    for k, v in miss_ids.items():
        reduced[k] = [x-date_range_ids[0] for x in v if x in date_range_ids]
    return reduced


def get_taylor_fig(ground_truth, loo_reconstruction, variable, start_date, end_date):
    station_indx_map = get_station_indices_map()
    gt_stations = extract_stations_from_nc(ground_truth, station_indx_map)
    pred_stations = extract_stations_from_nc(loo_reconstruction, station_indx_map)

    missing_indicies = get_nan_ids(gt_stations)
    deseas_pred_stations = handle_saisonality_on_ta(pred_stations)  # Pred has no NaNs
    deseas_gt_stations = handle_saisonality_on_ta(gt_stations, missing_indicies)  # GT has NaNs

    missing_indicies = _remove_miss_ids_not_in_range(missing_indicies, start_date, end_date)
    deseas_pred_stations = _cut_station_dict_to_date_range(deseas_pred_stations, start_date, end_date)
    deseas_gt_stations = _cut_station_dict_to_date_range(deseas_gt_stations, start_date, end_date)

    taylor_metrics = get_loo_taylor_metrics(deseas_gt_stations, deseas_pred_stations, missing_indicies)

    if variable == "both":
        std_devs = [m['norm_std'] for m in taylor_metrics.values()]
        corrs = [m['corr'] for m in taylor_metrics.values()]
        labels = list(taylor_metrics.keys())
    else:
        tm_var = {k.split('_')[0]: v for k, v in taylor_metrics.items() if f"_{variable}" in k}
        std_devs = [m['norm_std'] for m in tm_var.values()]
        corrs = [m['corr'] for m in tm_var.values()]
        labels = list(tm_var.keys())

    fig = create_normed_taylor_diagram(ref_std=1,
                                       test_std_devs=std_devs,
                                       test_corrs=corrs,
                                       labels=labels,
                                       )
    return fig
