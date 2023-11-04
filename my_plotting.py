import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import grid_finder
from mpl_toolkits.axisartist import floating_axes


def display_predictions(pred_mat, first_day_date):

    dates = pd.date_range(start="1807-01-01", periods=365, freq="D")
    dates = pd.Series(dates)
    day_id = dates.index[dates.dt.date == first_day_date].values[0]

    fig, axs = plt.subplots(2,5, figsize=(16,4))

    min_ta = np.min(pred_mat[day_id:day_id+5,...,0])
    max_ta = np.max(pred_mat[day_id:day_id+5,...,0])

    min_slp = np.min(pred_mat[day_id:day_id+5,...,1])
    max_slp = np.max(pred_mat[day_id:day_id+5,...,1])

    for i in range(5):
        ms_ta = axs[0,i].matshow(pred_mat[day_id+i,...,0], vmin=min_ta, vmax=max_ta)
        ms_slp = axs[1,i].matshow(pred_mat[day_id+i,...,1], vmin=min_slp, vmax=max_slp)

        axs[0, i].set_title(f"{dates.loc[day_id+i].strftime('%Y-%m-%d')}", fontsize=16)

        if i == 4:
            #                      [left, bottom, width, height]
            sub_ax1 = fig.add_axes([0.91, 0.57, 0.01, 0.26])
            fig.colorbar(ms_ta, cax=sub_ax1)
            sub_ax2 = fig.add_axes([0.91, 0.15, 0.01, 0.26])
            fig.colorbar(ms_slp, cax=sub_ax2)
    return fig


def create_normed_taylor_diagram(*, ref_std, test_std_devs, test_corrs, labels, fig=None):
    """
    Creates Taylor Diagram.
    :param ref_std: Reference standard deviation
    :param test_std_devs: Standard deviations of test (prediction) data.
    :param test_corrs: Correlations of prediction to ground truth data.
    :param labels: Labels to use in legend.
    :param fig: Figure to use, if already exists.
    :return:
    """

    if not (len(test_std_devs) == len(test_corrs) == len(labels)):
        raise Exception("Test standard deviations, correlations and labels must have the same length!")

    polar_transform = PolarAxes.PolarTransform()

    # Correlation
    corr_labels = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
    polar_axis_min = 0
    polar_axis_max = np.pi/2
    polar_label_locations = np.arccos(corr_labels)
    corr_pos_label_mapper = grid_finder.DictFormatter({
        polar_label_locations[i] : str(corr_labels[i]) for i in range(len(corr_labels))
    })
    locator = grid_finder.FixedLocator(polar_label_locations)

    # Standard deviation
    std_axis_min = 0
    std_axis_max = 2 * ref_std  # TODO: pass as arg?

    grid_helper = floating_axes.GridHelperCurveLinear(aux_trans=polar_transform,
                                                      extremes=(polar_axis_min, polar_axis_max,
                                                                std_axis_min, std_axis_max),
                                                      grid_locator1=locator,
                                                      tick_formatter1=corr_pos_label_mapper)

    if not fig:
        fig = plt.figure(figsize=(12,10))

    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax)

    # Add Grid
    ax.grid()

    # Adjust axes
    ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")

    ax.axis["left"].set_axis_direction("bottom")  # "X axis"
    ax.axis["left"].label.set_text("Standard deviation")

    ax.axis["right"].set_axis_direction("top")    # "Y-axis"
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")

    ax.axis["bottom"].set_visible(False)

    # Add Reference Point
    polar_ax = ax.get_aux_axes(polar_transform)
    polar_ax.plot([0], ref_std, "or", markersize=6, label="Ref")

    # Add STD-reference line
    std_ref_line_x = np.linspace(0, polar_axis_max)
    std_ref_line_y = np.zeros_like(std_ref_line_x) + ref_std
    polar_ax.plot(std_ref_line_x, std_ref_line_y, 'k:')

    # Add RMSE contour lines
    rmse_a, rmse_b = np.meshgrid(np.linspace(std_axis_min, std_axis_max),
                                 np.linspace(polar_axis_min, polar_axis_max))
    # According to the law of cosine:
    rmse_ = np.sqrt(ref_std**2 + rmse_a**2 - 2*ref_std*rmse_a*np.cos(rmse_b))
    contour_set = polar_ax.contour(rmse_b, rmse_a, rmse_, levels=4, colors='black', linestyles='--')
    plt.clabel(contour_set, inline=1, fontsize=10, colors='black')


    # Plot samples
    nbr_samples = len(test_std_devs)
    colors = plt.matplotlib.cm.jet(np.linspace(0, 1, nbr_samples))
    for i in range(nbr_samples):
        polar_ax.plot(np.arccos(test_corrs[i]), test_std_devs[i], 'x', label=labels[i],
                      color=colors[i],
                      markersize=10)

    #plt.legend(prop=dict(size='small'), loc='best')
    plt.legend(prop=dict(size='small'), loc='center left', ncols=2, bbox_to_anchor=(1, 0.5))

    return fig


def create_station_line_plot(pred, gt, stat_id):
    fig, ax = plt.subplots(1, 1)

    ax.plot(pred, 'red', label="Reconstruction")
    ax.plot(gt, 'blue', label="Station observation")

    ax.set_ylabel("ta [°C]" if "_ta" in stat_id else "slp [Pa]")
    ax.set_xlabel("Days")

    ax.grid(True)
    ax.legend()

    return fig
