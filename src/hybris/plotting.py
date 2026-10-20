import pandas as pd
import matplotlib as plt
import numpy as np

######### VISUALIZATION #########
def plot_hybris(hybris, plot_tillages = True, plot_dormant=False, add_groundtruth = True, add_pred_sow_harv=True, ax = None, date_col = 'date', color = 'brown', legend = True, alpha_smooth = 1, alpha_raw = 0.3):
    """Plot a HyBRIS series with optional events and dormant periods.

    Parameters
    ----------
    hybris : pandas.DataFrame
        Series containing the selected date column and HyBRIS columns. Event
        overlays use ``pred_type``, ``obs_type``, and ``in_growing_season``.
    plot_tillages, plot_dormant, add_groundtruth, add_pred_sow_harv : bool
        Enable the corresponding prediction, shading, observation, and
        phenology overlays.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new 12-by-5 inch axes is created otherwise.
    date_col : str, default 'date'
        Date column to plot.
    color : str, default 'brown'
        Color for the raw and smoothed series.
    legend : bool, default True
        Whether to add an outside legend.
    alpha_smooth, alpha_raw : float
        Opacity of the smoothed and raw lines.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        
    # Plot smoothed time series
    ax.plot(hybris[date_col], hybris["daily_index_smooth"], label="Smoothed", color=color, alpha = alpha_smooth)

    # Plot raw time series with transparency
    if "daily_index" in hybris.columns:
        ax.plot(hybris[date_col], hybris["daily_index"], label="Unsmoothed", color=color, alpha=alpha_raw)

    
        # Handle tillage points
        if plot_tillages:
            if "pred_tillage" in hybris["pred_type"].unique():
                tillage_all = hybris[hybris["pred_type"] == "pred_tillage"]
            
                #Plot other tillages
                ax.scatter(tillage_all[date_col], tillage_all["Value"],
                        label="Predicted tillage",
                        color="brown", marker="s", zorder=5)
    
    # Add groundtruth events as vertical lines
    if add_groundtruth:
        if hybris is not None:
            category_map = {"obs_tillage": 'obs. tillage',
                            'obs_sowing': 'obs. sowing',
                            'obs_harvest': 'obs. harvest'
                               }
            category_colors = {"obs_tillage": 'brown',
                                'obs_sowing': 'green',
                                'obs_harvest': 'black'
                                  }
            category_style = {"obs_tillage": '--',
                               'obs_sowing': '-.',
                               'obs_harvest': '-.'
                                 }

            added_labels = set()
            for category, color in category_colors.items():
                dates_to_mark = hybris.loc[hybris['obs_type'] == category, 'Date']
                for date in dates_to_mark:
                    label = category_map[category] if category not in added_labels else None
                    style = category_style[category]
                    ax.axvline(date, color=color, linestyle=style, alpha=0.8, label=label)
                    added_labels.add(category)

    # Plot phenological points if provided
    if add_pred_sow_harv:
        if hybris is not None:
            type_colors = {
                "pred_sowing": "green",
                "peak": "red",
                "pred_harvest": "black"
            }
            type_style = {"pred_sowing": 'o', 'peak': '^', 'pred_harvest': 'o'}
            lebel_typ = {"pred_sowing": 'Predicted sowing', 'peak': 'Peak', 'pred_harvest': 'Predicted harvest'}
            # Handle normal phenology (sowing, peak, harvest)
            for typ in ["pred_sowing", "peak", "pred_harvest"]:
                group = hybris[hybris["pred_type"] == typ]
                ax.scatter(group[date_col], group["Value"],
                        label=lebel_typ[typ],
                        color=type_colors.get(typ, "blue"),
                        marker=type_style[typ],
                        zorder=5)
            
    # Plot dormant periods (from harvest to next sowing)
    if plot_dormant:
        ymin, ymax = ax.get_ylim()          # vertical span of the plot
        ax.fill_between(hybris[date_col],
                        -0, 1,         # shade full height
                        where=~hybris["in_growing_season"],
                        color="gray", alpha=0.3,
                        label="Dormant period",
                        step="post")        # keeps the shading block‑wise

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    if legend:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )
    ax.grid(False)
    return ax


def plot_time_series(values, dates, groundtruth=None, season_boundaries=None,
                      tillages = None, tillage_windows = None, dormant_periods = None, color='gray',
                        title="Time Series", add = False, show = True, alpha = 1, marker = None,
                        ax = None, legend = True):
    """Plot a time series and optional observations, predictions, and windows.

    Ground-truth event codes are interpreted as 1=tillage, 2=sowing, and
    3=harvest. Optional season boundaries, tillage windows, and dormant
    periods are overlaid when their expected columns are present.

    Parameters:
    -----------
    values : array-like
        The values of the time series.

    dates : array-like
        The dates corresponding to the time series.

    groundtruth : pd.DataFrame, optional
        A DataFrame with event dates and categories.

    season_boundaries : pd.DataFrame, optional
        A DataFrame with 'SOS Date', 'EOS Date', and 'Peak Date' columns.

    color : str, optional (default='blue')
        Color of the time series line.

    title : str, optional (default="Time Series")
        Title of the plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    # Create new axes only if add is False and no ax is passed
    if add is False and ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # # If ax is still None (edge case), create one
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(12, 5))

    mask = ~np.isnan(values) & ~pd.isnull(dates)
    values = np.array(values)[mask]
    dates = np.array(dates)[mask]
    sorted_idx = np.argsort(dates)
    values = values[sorted_idx]
    dates = dates[sorted_idx]

    ax.plot(dates, values, label=title, color=color, alpha=alpha, marker=marker)

    # Add groundtruth events as vertical lines
    if groundtruth is not None:
        category_map = {1: 'Tillage', 2: 'Sowing', 3: 'Harvest'}
        category_colors = {1: 'brown', 2: 'green', 3: 'black'}
        category_style = {1: '--', 2: '-.', 3: '-.'}

        added_labels = set()
        for category, color in category_colors.items():
            dates_to_mark = groundtruth.loc[groundtruth['Ti.1_So.2_Ha.3'] == category, 'Date']
            for date in dates_to_mark:
                label = category_map[category] if category not in added_labels else None
                style = category_style[category]
                ax.axvline(date, color=color, linestyle=style, alpha=0.8, label=label)
                added_labels.add(category)

    # Add season boundaries (SOS, EOS, Peaks)
    if season_boundaries is not None:
        ax.scatter(season_boundaries['SOS Date'], season_boundaries['SOS Value'], 
                    color='green', marker='o', label='Predicted sowing')
        ax.scatter(season_boundaries['Peak Date'], season_boundaries['Peak Value'], 
                    color='red', marker='^', label='Peak')
        ax.scatter(season_boundaries['EOS Date'], season_boundaries['EOS Value'], 
                    color='black', marker='o', label='Predicted harvest')
        # Add season boundaries (SOS, EOS, Peaks)
    if tillages is not None:
        ax.scatter(tillages['Date'], tillages['Value'], 
                    color='brown', marker='o', label='Predicted tillage')
    if tillage_windows is not None:
        # Convert 'window_start' and 'window_end' to datetime
        tillage_windows.loc[:,'window_start'] = pd.to_datetime(tillage_windows['window_start'])
        tillage_windows.loc[:,'window_end'] = pd.to_datetime(tillage_windows['window_end'])
  
        # Iterate over each row in the dataframe and add vertical lines and shaded areas
        for index, row in tillage_windows.iterrows():
            window_start = row['window_start']
            window_end = row['window_end']
            
            # Add vertical lines at window start and window end
            ax.axvline(window_start, color='red', linestyle='--', lw=2, label='Window Start' if index == 0 else "")
            ax.axvline(window_end, color='blue', linestyle='--', lw=2, label='Window End' if index == 0 else "")
            
            # Add shaded area between window start and window end
            ax.axvspan(window_start, window_end, color='gray', alpha=0.3) 

    # Plot shaded dormant periods  
    if dormant_periods is not None:
        for i, row in dormant_periods.iterrows():
            eos = pd.to_datetime(row['Harvest Date'])
            sos = pd.to_datetime(row['Sowing Date'])
            ax.axvspan(eos, sos, color="gray", alpha=0.3, label="Dormant Period" if i == 0 else "")

    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5)) # (x, y) coordinates relative to the axes)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)
    if show:
        plt.show()

    return ax