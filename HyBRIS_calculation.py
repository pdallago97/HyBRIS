import pandas as pd
import numpy as np
from scipy.signal import find_peaks

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
