import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def normalize_percentiles(data, lower_percentile=0.02, upper_percentile=0.98):
    """
    Normalize a time series using the specified percentiles.
    
    Args:
        data (array-like): Input data to be normalized.
        lower_percentile (float): Lower percentile (default: 2nd percentile).
        upper_percentile (float): Upper percentile (default: 98th percentile).
        
    Returns:
        numpy array: Normalized data clipped between 0 and 1.
    """
    # Step 1: Calculate the specified percentiles
    percentiles = np.percentile(data[~np.isnan(data)], [lower_percentile * 100, upper_percentile * 100])
    
    # Step 2: Normalize using the calculated percentiles
    normalized_data = (data - percentiles[0]) / (percentiles[1] - percentiles[0])
    
    # Step 3: Clip values to the range [0, 1]
    normalized_data = np.clip(normalized_data, 0, 1)
    
    return normalized_data


def add_vis_radar(df):
    """
    Add Radar Vegetation Indices (RVI and others) to the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe with required radar bands (VH, VV).
        
    Returns:
        pandas.DataFrame: DataFrame with added RVI, VH_VV, VV_VH, and RI2 columns.
    """
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Check for required bands
    required_bands = ['VH', 'VV']
    for band in required_bands:
        if band not in df.columns:
            raise ValueError(f"Missing required band: {band}")
    
    # Ensure all bands are numeric
    for band in required_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # # Convert dB to linear scale
    # vv_lin = 10 ** (df['VV'] / 10)
    # vh_lin = 10 ** (df['VH'] / 10)

    # Calculate Radar Vegetation Indices
    with np.errstate(divide='ignore', invalid='ignore'):
        df['RVI'] = normalize_percentiles((4 * df['VH']) - (df['VV'] + df['VH']))
        #df["RVI"] = normalize_percentiles(4 * vh_lin / (vv_lin + vh_lin))
        df['VH_VV'] = normalize_percentiles(df['VH'] - df['VV'])
        df['VV_VH'] = normalize_percentiles(df['VV'] - df['VH'])
        df['RI2'] = normalize_percentiles((df['VV'] - df['VH']) / (df['VV'] + df['VH']))
    
    return df


def add_vis(df):
    """
    Add Vegetation Indices (VIs) to the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe with required bands (B2, B3, B4, B8, B11, B12).
        
    Returns:
        pandas.DataFrame: DataFrame with added NDVI, BSI, NDTI, NDSI, and NDWI columns.
    """
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Ensure no division by zero or invalid operations
    with np.errstate(divide='ignore', invalid='ignore'):
        # NDVI
        df['NDVI'] = normalize_percentiles((df['B8'] - df['B4']) / (df['B8'] + df['B4']))
        
        # BSI
        df['BSI'] = normalize_percentiles(((df['B11'] + df['B4']) - (df['B8'] + df['B2'])) / 
                                          ((df['B11'] + df['B4']) + (df['B8'] + df['B2'])))
        
        # NDTI
        df['NDTI'] = normalize_percentiles((df['B11'] - df['B12']) / (df['B11'] + df['B12']))
        
        # NDSI
        df['NDSI'] = normalize_percentiles((df['B3'] - df['B11']) / (df['B3'] + df['B11']))
        
        # NDWI
        df['NDWI'] = normalize_percentiles((df['B8'] - df['B12']) / (df['B8'] + df['B12']))

        #EVI
        df['EVI'] = 2.5 * (df['B8'] - df['B4']) / ((df['B8'] + 6 * df['B4'] - 7.5 * df['B2']) + 1)

    return df

def openSentinel2file(filename, bandsused):
    """
    Opens a Sentinel-2 CSV file, processes it, and returns a DataFrame.
    
    Parameters:
    filename (str): The path to the CSV file.
    column_names (list): The list of column names for the DataFrame.
    bandsused (list): The list of band columns to be processed.
    add_vis (function): A function to add visualization columns to the DataFrame.
    
    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    # Define column names
    column_names = ['date', 'ID'] + bandsused

    try:
        # Open the CSV file and assign column names
        df = pd.read_csv(filename, header=None, names=column_names)
        
        # Replace -9999 with NaN
        df.replace(-9999, pd.NA, inplace=True)
        #remove NA
        df = df.dropna()
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure bandsused columns are numeric and divide by 10,000
        df[bandsused] = df[bandsused].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors to NaN
        df[bandsused] = df[bandsused].div(10000)
        
        # Merge observation which happen on the same date by taking the maximum value
        df = df.groupby('date', as_index=False).max()

        return df

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        exit()

def openSentinel1file(filename, bandsusedS1):
    """
    Opens a Sentinel-1 CSV file, processes it, and returns a DataFrame.
    
    Parameters:
    filename (str): The path to the CSV file.
    bandsused (list): The list of band columns to be processed.
    add_vis (function): A function to add visualization columns to the DataFrame.
    
    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    # Define column names
    column_names = ['date', 'ID', 'orbit', 'orbitNumber'] + bandsusedS1

    # Open the CSV file and assign column names
    try:
        df = pd.read_csv(filename, header=None, names=column_names)

        # Replace -9999 with NaN
        df.replace(-9999, pd.NA, inplace=True)
        #remove NA
        df = df.dropna()

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure bandsused columns are numeric
        df[bandsusedS1] = df[bandsusedS1].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors to NaN

        # Merge observation which happen on the same date by taking the max value
        df = df.groupby('date', as_index=False).max()
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        exit()

def selectOrbit(df, selectMostPresent=False, selectHighestAngle=False):
    """
    Filters the Sentinel-1 dataframe by orbit type and orbit number.

    Parameters:
    df (pd.DataFrame): The Sentinel-1 time series dataframe.
    selectMostPresent (bool): If True, returns only the orbitNumber with the most observations.
    selectHighestAngle (bool): If True, returns only the orbitNumber with the highest average angle.

    Returns:
    dict (if both flags are False) or pd.DataFrame (if one flag is True)
    """
    grouped = df.groupby(['orbit', 'orbitNumber'])

    if selectMostPresent:
        # Orbit with the most observations
        most_present_orbit = max(grouped.groups, key=lambda x: len(grouped.get_group(x)))
        return grouped.get_group(most_present_orbit)

    if selectHighestAngle:
         # Orbit with the highest mean angle
        highest_angle_orbit = max(grouped.groups, key=lambda x: grouped.get_group(x)['angle'].mean())
        return grouped.get_group(highest_angle_orbit)

    # Return all orbit combinations as dictionary
    return {f"{orbit}_{orbitNum}": group for (orbit, orbitNum), group in grouped}


def daily_index_with_contributions(s2df, s1df, maxDiff=30, bandS2='NDVI', bandS1='RVI'):
    """
    Calculate a daily time series using the specified bands.
    
    Args:
        s2df (pd.dataframe): Input data from Sentinel 2. This dataframe should contain a 'date' column and the specified band.
        s1df (pd.dataframe): Input data from Sentinel 1. This dataframe should contain a 'date' column and the specified band.
        maxDiff (int): window size. It is the maximum time difference in days to consider for combining the two data (default: +-30 days).
        bandS2 (str): Band name for Sentinel-2 data (default: 'NDVI').
        bandS1 (str): Band name for Sentinel-1 data (default: 'RVI').
               
    Returns:
        pd.dataframe: Normalized time series which combines the bandS2 and the bandS1, clipped between 0 and 1.
        It also contains the contributions of each band (Sentinel 1 or Sentinel 2) to the daily index.
    """
        
    # Select band for Sentinel-2 and Sentinel-1
    s2band = s2df[bandS2].values
    s1band = s1df[bandS1].values

    # Create a sequence of daily dates covering the range of observations
    start_date = min(s1df['date'].min(), s2df['date'].min())
    end_date = max(s1df['date'].max(), s2df['date'].max())
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize the new dataframe
    daily_data = pd.DataFrame({'date': daily_dates, 'daily_index': np.nan, 's1_contribution': np.nan, 's2_contribution': np.nan})

    # Calculate the weighted mean and contributions for each day
    for i, current_date in enumerate(daily_data['date']):
        s1_time_diff = np.abs((current_date - s1df['date']).dt.days)
        s2_time_diff = np.abs((current_date - s2df['date']).dt.days)

        combined_values = np.concatenate([s1band, s2band])
        combined_time_diffs = np.concatenate([s1_time_diff, s2_time_diff])

        # Valid mask
        valid_mask = (~np.isnan(combined_values)) & (~np.isnan(combined_time_diffs)) & (combined_time_diffs <= maxDiff)
        
        if np.any(valid_mask):
            weights = 1 / (combined_time_diffs[valid_mask] + 1)

            n_s1 = len(s1band)
            # Split into Sentinel-1 and Sentinel-2
            s1_weights = weights[:np.sum((~np.isnan(s1band)) & (s1_time_diff <= maxDiff))]
            s2_weights = weights[np.sum((~np.isnan(s1band)) & (s1_time_diff <= maxDiff)):]
            
            s1_weight_sum = np.sum(s1_weights)
            s2_weight_sum = np.sum(s2_weights)
            total_weight_sum = s1_weight_sum + s2_weight_sum

            # Calculate daily index
            daily_data.loc[i, 'daily_index'] = np.average(combined_values[valid_mask], weights=weights)

            # Calculate contributions
            if total_weight_sum > 0:
                daily_data.loc[i, 's1_contribution'] = (s1_weight_sum / total_weight_sum)
                daily_data.loc[i, 's2_contribution'] = (s2_weight_sum / total_weight_sum)

    # Normalize daily index
    daily_data['daily_index'] = normalize_percentiles(daily_data['daily_index'])

    return daily_data


def calculate_hybris(s1, s2, bandS2 = 'BSI', bandS1 = 'VV_VH'):

    """
    Calculate the Hybris index using Sentinel-1 and Sentinel-2 data.
    Args:
        s1 (pd.DataFrame): Sentinel-1 data with columns 'date', 'VV_VH', and 'ID'.
        s2 (pd.DataFrame): Sentinel-2 data with columns 'date', 'BSI', and 'ID'.
        bandS2 (str): Band name for Sentinel-2 data (default: 'BSI').
        bandS1 (str): Band name for Sentinel-1 data (default: 'VV_VH').
    Returns:
        pd.DataFrame: A DataFrame containing the Hybris index, smoothed index, and ID.
    """

    #Calculate fused index using BSI and VV_VH
    hybris = daily_index_with_contributions(s2, s1, bandS2 = bandS2, bandS1= bandS1)

    #Invert index to resemble a vegetation index
    hybris['daily_index'] = 1 - hybris['daily_index']

    #Smooth the index
    hybris['daily_index_smooth'] = hybris['daily_index'].rolling(window=30, center=True, min_periods=1).mean()

    #Add ID column
    ID_field = s2['ID'].unique()[0]  # Extract the field ID from the Sentinel-2 data
    hybris['ID'] = ID_field  # Add the field ID to the daily index DataFrame

    #Add ID_date column
    hybris['ID_date'] = hybris['ID'].astype(str) + "_" + hybris['date'].dt.strftime('%Y-%m-%d')

    return pd.DataFrame(hybris)

def find_maxima(hybris, distancePOS=60, prominencePOS=(0.1, 1)):
    """
    Finds the maximum points (peaks) in a time series.

    Parameters:
    -----------
    hybris is a DataFrame containing the time series data with columns:
        time_series : array-like
            A time series of vegetation index values (e.g., NDVI, EVI, daily_index).
        
        dates : array-like
            Corresponding dates for the vegetation index values.
    
    distancePOS : int, optional (default=60)
        Minimum distance between detected Peak of Seasons.
    
    prominencePOS : tuple (default=(0.1, 1))
        Minimum prominence required for Peak of Seasons.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing maxima detected in the time series, with columns:
        - `Date`
        - `Value`
        - `Prominence`
    """
    
    # Detect all maxima in the time series
    peaks, peak_properties = find_peaks(hybris["daily_index_smooth"], distance=distancePOS, prominence=prominencePOS)
    peak_prominences = peak_properties["prominences"]

    # Create a DataFrame to store results
    maxima = []
    
    for i in range(len(peaks)):
        maxima.append({
            "Date": hybris["date"].iloc[peaks[i]],
            "Value": hybris["daily_index_smooth"][peaks[i]],
            "Prominence": peak_prominences[i]
        })

    return pd.DataFrame(maxima)

def find_minima(hybris, distanceMin=15, prominenceMin=(0.1, 1), distanceTillages=30, prominenceTillages=(0.1, 1), height = None):
    """
    Finds the minimum points (valleys) in a time series.

    Parameters:
    -----------
    hybris is a DataFrame containing the time series data with columns:
        time_series : array-like
            A time series of vegetation index values (e.g., NDVI, EVI, daily_index).
        
        dates : array-like
            Corresponding dates for the vegetation index values.
    
    distanceMin : int, optional (default=15)
        Minimum distance between detected minima.
    
    prominenceMin : tuple (default=(0.1, 1))
        Minimum prominence required for minima.

    distanceTillages : int, optional (default=1)
        Minimum distance between detected tillages.
    
    prominenceTillages : tuple (default=(0.1, 1))
        Minimum prominence required for tillages.

    height : float, optional (default=None)
        Minimum height of the minima to be detected. If None, no height filtering is applied.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing minima detected in the time series, with columns:
        - `Date`
        - `Value`
        - `Prominence`
    """
    ### SOWING
    # Detect all smooth minima in the time series (sowing)
    sowing, sow_properties = find_peaks(-hybris["daily_index_smooth"], distance=distanceMin, prominence=prominenceMin)
    sow_properties = sow_properties["prominences"]

    ### HARVEST
    # Detect all minima in the time series in rough time series (harvest)
    harvest, harv_properties = find_peaks(-hybris["daily_index"], distance=distanceMin, prominence=prominenceMin, height=height)
    harv_properties = harv_properties["prominences"]

    ### TILLAGE
    #Detect minima in the time series in rough time series (tillage)
    tillage, tillage_properties = find_peaks(-hybris["daily_index"], distance=distanceTillages, prominence=prominenceTillages, height=height)
    tillage_properties = tillage_properties["prominences"]

    ## store results into a datafrane
    minima = []

    # Add sowing
    for i, idx in enumerate(sowing):
        minima.append({
            "Date": hybris["date"].iloc[idx],
            "Value": hybris["daily_index_smooth"].iloc[idx],
            "Prominence": sow_properties[i],
            "Type": "sowing"
        })

    # Add harvest
    for i, idx in enumerate(harvest):
        minima.append({
            "Date": hybris["date"].iloc[idx],
            "Value": hybris["daily_index"].iloc[idx],
            "Prominence": harv_properties[i],
            "Type": "harvest"
        })

    # Add tillage
    for i, idx in enumerate(tillage):
        minima.append({
            "Date": hybris["date"].iloc[idx],
            "Value": hybris["daily_index"].iloc[idx],
            "Prominence": tillage_properties[i],
            "Type": "tillage"
        })

    # Convert to DataFrame
    minima = pd.DataFrame(minima)
    return pd.DataFrame(minima)

    
def growing_seasons(maxima, minima):
    """    Create a DataFrame with growing seasons based on maxima and minima.
    Args:
        maxima (pd.DataFrame): DataFrame containing maxima with columns 'Date', 'Value', and 'Prominence'.
        minima (pd.DataFrame): DataFrame containing minima with columns 'Date', 'Value', 'Prominence', and 'Type'.
    Returns:
        pd.DataFrame: A DataFrame with growing seasons, including sowing, peak, and harvest events.
    """
    # Ensure maxima and minima are sorted by date
    # Filter only sowing and harvest events
    sowings = minima[minima["Type"] == "sowing"].sort_values("Date")
    harvests = minima[minima["Type"] == "harvest"].sort_values("Date")

    results = []

    for i, row in maxima.iterrows():
        season_id = i + 1
        peak_date = row["Date"]
        peak_value = row["Value"]
        peak_prom = row["Prominence"]

        has_sowing = False
        has_harvest = False

        # Find the last sowing before the peak
        sowing_before_peak = sowings[sowings["Date"] < peak_date]
        if not sowing_before_peak.empty:
            sowing = sowing_before_peak.iloc[-1]
            has_sowing = True
            sowing_date = sowing["Date"]
            results.append({
                "Date": sowing["Date"],
                "Value": sowing["Value"],
                "Prominence": sowing["Prominence"],
                "pred_type": "pred_sowing",
                "season_id": season_id,
                "season_complete": None,  # will be filled later
                "season_length": None     # will be filled later
            })
        else:
            # Add placeholder with NaN
            results.append({
                "Date": pd.NaT,
                "Value": None,
                "Prominence": None,
                "pred_type": "pred_sowing",
                "season_id": season_id,
                "season_complete": None,
                "season_length": None     # will be filled later
            })

        # Add the peak
        results.append({
            "Date": peak_date,
            "Value": peak_value,
            "Prominence": peak_prom,
            "pred_type": "peak",
            "season_id": season_id,
            "season_complete": None,
            "season_length": None     # will be filled later
        })

        # Find the first harvest after the peak
        harvest_after_peak = harvests[harvests["Date"] > peak_date]
        if not harvest_after_peak.empty:
            harvest = harvest_after_peak.iloc[0]
            has_harvest = True
            harvest_date = harvest["Date"]
            results.append({
                "Date": harvest["Date"],
                "Value": harvest["Value"],
                "Prominence": harvest["Prominence"],
                "pred_type": "pred_harvest",
                "season_id": season_id,
                "season_complete": None,
                "season_length": None     # will be filled later
            })
        else:
            # Add placeholder with NaN
            results.append({
                "Date": pd.NaT,
                "Value": None,
                "Prominence": None,
                "pred_type": "pred_harvest",
                "season_id": season_id,
                "season_complete": None,
                "season_length": None     # will be filled later
            })

        # Set season_complete flag for all three rows of this season
        complete = has_sowing and has_harvest

        # Calculate season length if complete
        if complete == True:
            season_length = (harvest_date - sowing_date).days
        else:
            season_length = pd.NaT

        for j in range(3):
            results[-1 - j]["season_complete"] = complete
            results[-1 - j]["season_length"] = season_length


   
    df_long = pd.DataFrame(results).sort_values(["season_id", "Date"])

    return df_long

def filter_small_seasons(df_long, min_season_length = 60, min_peak_value = 0.4):
    # Filter out seasons and with duration < 60 days and peak < 0.4
    # We exclude these as they are unlikely to be a crop or a cover crop.
    # These are ASSOCIATED WITH WEED REGROWTH
    invalid_seasons = df_long[
        (df_long["season_complete"] == True) &  # Only consider complete seasons
        ((df_long["season_length"] < min_season_length) & (df_long["Value"] < min_peak_value)) &  # Filter by season length or peak value
        (df_long["pred_type"] == "peak")  # Only consider rows of type 'peak'
    ]["season_id"].unique()

    df_long = df_long[~df_long["season_id"].isin(invalid_seasons)]

    return df_long

def add_tillages(g_seasons, minima):
    """
    Flag tillage events based on the growing seasons and minima.
    """
    # Filter tillages
    tillage_df = minima[minima["Type"] == "tillage"].copy().rename(columns={"Type": "pred_type"})
    tillage_df["pred_type"] = "pred_tillage"

    # Prepare coincidence check
    phen_dates = g_seasons[g_seasons["pred_type"].isin(["pred_sowing", "pred_harvest"])]["Date"]
    tillage_df["coincident"] = tillage_df["Date"].isin(phen_dates)

    # Merge into hybris
    results = pd.concat([g_seasons, tillage_df], ignore_index=True).sort_values(["Date"])
    return results

def add_predictions(hybris, predictions):
    """
    Add predictions to the hybris DataFrame and assign growing season flags.
    Args:
        hybris (pd.DataFrame): DataFrame containing the Hybris index with a 'date' column.
        predictions (pd.DataFrame): DataFrame containing predicted events with a 'date' column and 'pred_type'.
    Returns:
        pd.DataFrame: A DataFrame with predictions merged into the Hybris index, including growing season flags.
    """

    hybris = hybris.copy().rename(columns={"date": "Date"})
    # Merge hybris time series with predicted events
    predictions = pd.merge(hybris, predictions, on="Date", how="left")

    #Add a flag True or False to assign to each date: is it part of a growing season or not?
    predictions = predictions.sort_values("Date").reset_index(drop=True)

    # Initialize column
    predictions["in_growing_season"] = False

    # Get event indices
    sowings = predictions.index[predictions["pred_type"] == "pred_sowing"]
    peaks   = predictions.index[predictions["pred_type"] == "peak"]
    harvests = predictions.index[predictions["pred_type"] == "pred_harvest"]

    # Mark sowing → peak
    for sow in sowings:
        peak_after = peaks[peaks > sow]
        if len(peak_after):
            predictions.loc[sow : peak_after[0], "in_growing_season"] = True

    # Mark peak → harvest
    for peak in peaks:
        harvest_after = harvests[harvests > peak]
        if len(harvest_after):
            predictions.loc[peak : harvest_after[0], "in_growing_season"] = True


    # #Filter out tillage predictions which are coincident with an harvest or a sowing event
    # #Filter out tillage predicions which are inside a growing season
    mask = (
        (predictions["pred_type"] == "pred_tillage") &
        ((predictions["coincident"] == True) | (predictions["in_growing_season"] == True))
    )

    predictions.loc[mask, "pred_type"] = 'pred_tillage_excluded'

    # assign season number to each row, based on the column 'in_growing_season'
    predictions['season_number'] = (
        predictions['in_growing_season']            # current value
        .ne(predictions['in_growing_season'].shift())  # True when the value changes
        .cumsum()                        # running total of the changes
    )

    return predictions

def merge_with_GT(merged, ground_truth): 
    # Prep inputs
    merged = merged.copy().rename(columns={"date": "Date"})
    ground_truth = ground_truth.copy()


    merged["Farm"] = ground_truth["Farm"].unique()[0]
    merged["Country"] = ground_truth["Country"].unique()[0]

    # Add event types
    type_map = {1: "obs_tillage", 2: "obs_sowing", 3: "obs_harvest"}
    ground_truth["obs_type"] = ground_truth["Ti.1_So.2_Ha.3"].map(type_map)

    # All combinations of merged dates and ground truth events on same date
    merged = merged.merge(
        ground_truth[["Date", "obs_type", "Crop", "Crop.type", "Method_EN", "Ti.1_So.2_Ha.3"]],
        on="Date",
        how="left"
    )
    return merged

def filter_by_date(df, start_date, end_date, date_column='date'):
    """
    Filters a DataFrame based on a date range.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'date' column.
        start_date (str or pd.Timestamp): Start date (inclusive).
        end_date (str or pd.Timestamp): End date (inclusive).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df[date_column] = pd.to_datetime(df[date_column])  # Ensure 'date' is datetime type
    return df[(df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))]

######### VISUALIZATION #########

def plot_hybris(hybris, plot_tillages = True, plot_dormant=False, add_groundtruth = True, add_pred_sow_harv=True, ax = None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        
    # Plot smoothed time series
    ax.plot(hybris["Date"], hybris["daily_index_smooth"], label="Smoothed", color="brown")

    # Plot raw time series with transparency
    if "daily_index" in hybris.columns:
        ax.plot(hybris["Date"], hybris["daily_index"], label="Unsmoothed", color="brown", alpha=0.3)

    
        # Handle tillage points
        if plot_tillages:
            if "pred_tillage" in hybris["pred_type"].unique():
                tillage_all = hybris[hybris["pred_type"] == "pred_tillage"]
            
                #Plot other tillages
                ax.scatter(tillage_all["Date"], tillage_all["Value"],
                        label="Predicted tillage",
                        color="brown", marker="s", zorder=5)
    
    # Add groundtruth events as vertical lines
    if add_groundtruth:
        if hybris is not None:
            category_map = {"obs_tillage": 'Tillage',
                            'obs_sowing': 'Sowing',
                            'obs_harvest': 'Harvest'
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
                ax.scatter(group["Date"], group["Value"],
                        label=lebel_typ[typ],
                        color=type_colors.get(typ, "blue"),
                        marker=type_style[typ],
                        zorder=5)
            
    # Plot dormant periods (from harvest to next sowing)
    if plot_dormant:
        ymin, ymax = ax.get_ylim()          # vertical span of the plot
        ax.fill_between(hybris["Date"],
                        -0, 1,         # shade full height
                        where=~hybris["in_growing_season"],
                        color="gray", alpha=0.3,
                        label="Dormant period",
                        step="post")        # keeps the shading block‑wise

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.legend(
    loc='upper left',        # Position relative to bbox
    bbox_to_anchor=(1.0, 0.8) # (x, y) coordinates relative to the axes
    )
    ax.grid(False)
   
    return ax


def plot_time_series(values, dates, groundtruth=None, season_boundaries=None,
                      tillages = None, tillage_windows = None, dormant_periods = None, color='gray',
                        title="Time Series", add = False, show = True, alpha = 1, marker = None,
                        ax = None):
    """
    Plots a time series and optionally adds vertical lines for events in the groundtruth DataFrame.
    It also overlays detected Start of Season (SOS), End of Season (EOS), and Peak points.

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

    Returns:
    --------
    None
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

    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5)) # (x, y) coordinates relative to the axes)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)
    if show:
        plt.show()

    return ax