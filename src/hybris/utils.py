import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import ee #to use the Google Earth Engine library, you need to authenticate and initialize it first (ee.Initialize()/ee.Authenticate())
import gee_s1_ard.wrapper as wp #Refer to Mulissa et al. 2021 https://doi.org/10.3390/rs13101954 and clone the repository from https://github.com/adugnag/gee_s1_ard/tree/main/python-api
import os

def getID(df, id_value, id_column='ID'):
    """
    Filters a DataFrame to return only rows where the specified column matches the given ID.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    id_value (any): The ID value to filter by.
    id_column (str): The column name to filter on (default is 'ID').
    
    Returns:
    pd.DataFrame: A DataFrame containing only the matching rows.
    """
    return df[df[id_column] == id_value]

def get_S2_one_field(field, field_id, bands, start_date, end_date, cloud_filter, output_dir, download = True):
    field = field.first()
    # Initialize cloud score collection
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    csPlusBands = csPlus.first().bandNames()

    def maskLowQA(image):
        qaBand = 'cs'
        clearThreshold = 0.5
        mask = image.select(qaBand).gte(clearThreshold)
        return image.updateMask(mask)

    # Initialize Sentinel-2 collection
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(field.geometry())
                .filterDate(start_date, end_date)
                .select(bands)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
                .linkCollection(csPlus, csPlusBands)
                .map(maskLowQA))
    
    def medianBands(image):
        medianValues = (image
                        .select(bands)
                        .reduceRegion(
                            reducer=ee.Reducer.median(), 
                            geometry=field.geometry(),
                            scale=10,
                            maxPixels=1e8
                        )
                        .combine({band: -9999 for band in bands}, overwrite=False))
        formattedValues = [medianValues.getNumber(band).format('%.4f') for band in bands]
        date = image.date().format('YYYY-MM-dd')
        output = [date, field_id] + formattedValues
        return image.set('output', output)

    timeSeries = (collection.map(medianBands))
    result = timeSeries.aggregate_array('output').getInfo()

    if download:
        filename = os.path.join(output_dir, f'Sentinel2_{field_id}.csv')
        with open(filename, 'w') as out_file:
            for items in result:
                line = ','.join([str(item) for item in items])
                print(line, file=out_file)
        print("Downloaded Sentinel 2")
        return filename
    else:
        return result
    
def get_S1_one_field(field, field_id, bandsusedS1, start_date, end_date, output_dir, download = True):
    
    field = field.first()

    collection = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                .filterDate(start_date, end_date)
                .select(bandsusedS1)
                .filterBounds(field.geometry())
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.eq('resolution_meters', 10))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    
    # Define parameters for Sentinel-1 preprocessing
    parameter = {
        'COLLECTION': collection,
        'START_DATE': start_date,
        'STOP_DATE': end_date,
        'POLARIZATION': 'VVVH',
        'ORBIT': 'BOTH',
        'ROI': field.geometry(),
        'APPLY_BORDER_NOISE_CORRECTION': True,
        'APPLY_SPECKLE_FILTERING': True,
        'SPECKLE_FILTER_FRAMEWORK': 'MONO',
        'SPECKLE_FILTER': 'LEE',
        'SPECKLE_FILTER_KERNEL_SIZE': 5,
        'SPECKLE_FILTER_NR_OF_IMAGES': 5,
        'APPLY_TERRAIN_FLATTENING': True,
        'DEM': ee.Image('USGS/SRTMGL1_003'),
        'TERRAIN_FLATTENING_MODEL': 'VOLUME',
        'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
        'FORMAT': 'DB',
        'CLIP_TO_ROI': False,
        'SAVE_ASSET': False,
        'ASSET_ID': "users/your_username/your_asset_name"  # Replace with your desired asset path
    }

    # Preprocessed Sentinel-1 collection
    collection = wp.s1_preproc(parameter)
    
    #calculate median
    def medianBands_S1(image):
        # Reduce the region for all bands at once
        medianValues = (image
                        .select(bandsusedS1)  # Select only the bands of interest
                        .reduceRegion(
                            reducer=ee.Reducer.median(),
                            geometry=field.geometry(),
                            scale=10,  # Adjust the scale as needed
                            maxPixels=1e8  # Increase if working with large geometries
                        )
                        .combine({band: -9999 for band in bandsusedS1}, overwrite=False))  # Set default values

        # Format median values for output
        formattedValues = [medianValues.getNumber(band).format('%.4f') for band in bandsusedS1]

        # Add the date
        date = image.date().format('YYYY-MM-dd')

        # Add the Sentinel 1 specifics
        orbit = image.get('orbitProperties_pass')
        orbitNumber = ee.Number(image.get('relativeOrbitNumber_start')) 
                                                            
        # Combine all output: [date, fieldID, band1_median, band2_median, ...]
        output = [date, field_id, orbit, orbitNumber] + formattedValues

        return image.set('output', output)
    
    timeSeries = (collection.map(medianBands_S1))
    
    result = timeSeries.aggregate_array('output').getInfo()

    if download:
        filename = os.path.join(output_dir, f'Sentinel1_{field_id}.csv')
        with open(filename, 'w') as out_file:
            for items in result:
                line = ','.join([str(item) for item in items])
                print(line, file=out_file)
        print("Done with Sentinel 1")
        return filename
    else:
        return result 

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
        pandas.DataFrame: DataFrame with added RVI, VH_VV, VV_VH columns.
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
    
    # Convert dB to linear scale
    vv_lin = 10 ** (df['VV'] / 10)
    vh_lin = 10 ** (df['VH'] / 10)

    # Calculate Radar Vegetation Indices
    with np.errstate(divide='ignore', invalid='ignore'):
        df['RVI'] = normalize_percentiles((4 * vh_lin)/(vv_lin + vh_lin)) #calculated using the linearly transformed VV and VH
        df['VH_VV'] = normalize_percentiles(df['VH'] - df['VV']) #calculated using the difference: VV and VH are in log scale
        df['VV_VH'] = normalize_percentiles(df['VV'] - df['VH']) #calculated using the difference: VV and VH are in log scale
    
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
        df['EVI'] = normalize_percentiles(2.5 * (df['B8'] - df['B4']) / ((df['B8'] + 6 * df['B4'] - 7.5 * df['B2']) + 1))

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

def daily_index_with_contributions_vectorized(
        s2df: pd.DataFrame,
        s1df: pd.DataFrame,
        maxDiff: int = 12,
        bandS2: str = 'BSI',
        bandS1: str = 'VH_VV',
        smoothing: int = 30
    ) -> pd.DataFrame:
        """
        Vectorized computation of daily_index, s1_contribution, s2_contribution.
        Returns a DataFrame with columns: date, daily_index, s1_contribution, s2_contribution, daily_index_smooth.
        """

        # --- Prepare date range ---
        # Ensure date columns are datetime64[ns]
        s1 = s1df.copy()
        s2 = s2df.copy()
        s1['date'] = pd.to_datetime(s1['date'])
        s2['date'] = pd.to_datetime(s2['date'])

        start_date = min(s1['date'].min(), s2['date'].min())
        end_date = max(s1['date'].max(), s2['date'].max())
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Convert to numpy datetime64[D] for integer day differences
        daily_np = daily_dates.values.astype('datetime64[D]')
        s1_dates_np = s1['date'].values.astype('datetime64[D]')
        s2_dates_np = s2['date'].values.astype('datetime64[D]')

        # --- Sensor values as numpy arrays ---
        s1_vals = s1[bandS1].values.astype(float) if bandS1 in s1.columns else np.full(len(s1), np.nan)
        s2_vals = s2[bandS2].values.astype(float) if bandS2 in s2.columns else np.full(len(s2), np.nan)

        # If either sensor has zero rows, handle gracefully
        n_days = daily_np.shape[0]
        n_s1 = s1_dates_np.shape[0]
        n_s2 = s2_dates_np.shape[0]

        # Edge: if no s1 or s2 rows, create empty arrays with correct shapes
        if n_s1 == 0:
            s1_dates_np = np.array([], dtype='datetime64[D]')
            s1_vals = np.array([], dtype=float)
        if n_s2 == 0:
            s2_dates_np = np.array([], dtype='datetime64[D]')
            s2_vals = np.array([], dtype=float)

        # --- Compute absolute day differences matrices (shape: n_days x n_sensor) ---
        # If sensor has zero rows, create empty (n_days x 0) arrays
        if s1_dates_np.size > 0:
            # broadcasting: (n_days,1) - (1,n_s1) -> (n_days,n_s1)
            s1_diff = np.abs((daily_np[:, None] - s1_dates_np[None, :]).astype('timedelta64[D]').astype(int))
        else:
            s1_diff = np.zeros((n_days, 0), dtype=int)

        if s2_dates_np.size > 0:
            s2_diff = np.abs((daily_np[:, None] - s2_dates_np[None, :]).astype('timedelta64[D]').astype(int))
        else:
            s2_diff = np.zeros((n_days, 0), dtype=int)

        # --- Valid masks (within maxDiff and not NaN) ---
        if s1_vals.size > 0:
            s1_notnan = ~np.isnan(s1_vals)  # shape (n_s1,)
            s1_valid_mask = (s1_diff <= maxDiff) & s1_notnan[None, :]
        else:
            s1_valid_mask = np.zeros((n_days, 0), dtype=bool)

        if s2_vals.size > 0:
            s2_notnan = ~np.isnan(s2_vals)
            s2_valid_mask = (s2_diff <= maxDiff) & s2_notnan[None, :]
        else:
            s2_valid_mask = np.zeros((n_days, 0), dtype=bool)

        # --- Weights: 1 / (days_diff + 1) where valid, else 0 ---
        # Avoid division by zero: diff >= 0 so +1 is safe
        if s1_diff.size > 0:
            s1_weights = np.where(s1_valid_mask, 1.0 / (s1_diff + 1), 0.0)
        else:
            s1_weights = np.zeros((n_days, 0), dtype=float)

        if s2_diff.size > 0:
            s2_weights = np.where(s2_valid_mask, 1.0 / (s2_diff + 1), 0.0)
        else:
            s2_weights = np.zeros((n_days, 0), dtype=float)

        # --- Weighted sums and contributions ---
        # Numerator: sum(weights * values) across sensor observations
        if s1_vals.size > 0:
            s1_vals_row = s1_vals[None, :]  # shape (1, n_s1)
            s1_num = (s1_weights * s1_vals_row).sum(axis=1)  # shape (n_days,)
            s1_wsum = s1_weights.sum(axis=1)
        else:
            s1_num = np.zeros(n_days, dtype=float)
            s1_wsum = np.zeros(n_days, dtype=float)

        if s2_vals.size > 0:
            s2_vals_row = s2_vals[None, :]
            s2_num = (s2_weights * s2_vals_row).sum(axis=1)
            s2_wsum = s2_weights.sum(axis=1)
        else:
            s2_num = np.zeros(n_days, dtype=float)
            s2_wsum = np.zeros(n_days, dtype=float)

        total_weight = s1_wsum + s2_wsum

        # Avoid division by zero: where total_weight == 0 set daily_index to nan
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_index_arr = (s1_num + s2_num) / total_weight
            daily_index_arr = np.where(total_weight > 0, daily_index_arr, np.nan)

            s1_contrib_arr = np.where(total_weight > 0, s1_wsum / total_weight, np.nan)
            s2_contrib_arr = np.where(total_weight > 0, s2_wsum / total_weight, np.nan)

        # --- Build DataFrame ---
        daily_data = pd.DataFrame({
            'date': daily_dates,
            'daily_index': daily_index_arr,
            's1_contribution': s1_contrib_arr,
            's2_contribution': s2_contrib_arr
        })

        # --- Normalize and smooth ---
        # normalize_percentiles must accept a pandas Series and return a Series/array of same length
        daily_data['daily_index'] = normalize_percentiles(daily_data['daily_index'])

        # Smooth the index (same as original)
        daily_data['daily_index_smooth'] = daily_data['daily_index'].rolling(window=smoothing, center=True, min_periods=1).mean()

        return daily_data

def calculate_hybris_vectorized(s1, s2, bandS2 = 'BSI', bandS1 = 'VV_VH', maxDiff=12): 
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
    hybris = daily_index_with_contributions_vectorized(s2, s1, bandS2 = bandS2, bandS1= bandS1, maxDiff=maxDiff)

    if bandS2 == 'BSI':
        #Invert index to resemble a vegetation index
        hybris['daily_index'] = 1 - hybris['daily_index']
        #Invert index to resemble a vegetation index
        hybris['daily_index_smooth'] = 1 - hybris['daily_index_smooth']

    #Add ID column
    ID_field = s2['ID'].unique()[0]  # Extract the field ID from the Sentinel-2 data
    hybris['ID'] = ID_field  # Add the field ID to the daily index DataFrame

    #Add ID_date column
    hybris['ID_date'] = hybris['ID'].astype(str) + "_" + hybris['date'].dt.strftime('%Y-%m-%d')

    return pd.DataFrame(hybris)

# Function to process a single field to store minima dates and ground truth dates
def wrapper_optical_band(s2_path, bandsused, s2_band, gt):
    
    # Open Sentinel-2 and Sentinel-1 files
    s2 = add_vis(openSentinel2file(s2_path, bandsused))

    # Create a daily date range spanning the entire dataset
    daily_dates = pd.DataFrame({"date": pd.date_range(start=s2["date"].min(), end=s2["date"].max(), freq="D")})

    # Merge with an outer join to keep all observations
    s2 = pd.merge(s2, daily_dates, on="date", how="outer")
   
    if s2_band == 'BSI':
        s2[s2_band] = 1 - s2[s2_band] # Invert the index for BSI

    # Retrieve management data for this field
    id = int(s2['ID'].unique()[0])
    field = getID(gt, id)

    ########### USING THE SAME LOGIC FOR HYBRIS TO ONE BAND DATASETS
    # Interpolate missing values using linear interpolation
    s2["daily_index"] = (s2[s2_band]
        .interpolate(method="linear")  # Fill missing values
    )

    s2['daily_index_smooth'] = s2["daily_index"].rolling(window=30, center=True, min_periods=1).mean()  # Apply rolling average
    
    #Find maxima and minima in the fused time series
    maxima = find_maxima(s2) #Peaks of seasons
    minima = find_minima(s2, prominenceMin = 0.1, distanceTillages=30, prominenceTillages=(0,1)) #Sowing, harvest, tillage

    #Identify growing seasons based on predicted minima
    g_seasons = growing_seasons(maxima, minima) #assign sowing and harvest dates to a peak of season

    #Add tillage predictions
    predictions = add_tillages(g_seasons, minima)

    #Check if field exists
    if field.empty:
        print(f"Field ID {id} not found in ground truth data.")

    #If field exists, proceed with analysis    
    else:

        #Merge with predictions
        s2 = add_predictions(s2, predictions)
        
        # Convert Date columns to datetime format
        field.loc[:, 'Date'] = pd.to_datetime(field['Date'], errors='coerce')

        s2 = merge_with_GT(s2, field)
        s2["ID"] = int(s2['ID'].unique()[0])

        results = validate_predictions(s2)

        return results, s2


def wrapper_radar_band(s1_path, bandsusedS1, s1_band, gt):

        s1 = openSentinel1file(s1_path, bandsusedS1)

        # Calculate daily index with both S1 orbits separately
        s1_asc = s1[s1['orbit'] == 'ASCENDING']  # Select only ascending orbits
        s1_des = s1[s1['orbit'] == 'DESCENDING']  # Select only descending orbits

        s1_des = selectOrbit(s1_des, selectMostPresent=True) # Select only most present orbit
        s1_asc = selectOrbit(s1_asc, selectMostPresent=True) # Select only most present orbit

        # Use both S1 orbits
        s1_des = add_vis_radar(s1_des)
        s1_asc = add_vis_radar(s1_asc)
        s1 = pd.concat([s1_des, s1_asc], ignore_index=True) 

        # Retrieve management data for this field
        id = int(s1['ID'].unique()[0])
        field = getID(gt, id)

        ########### USING THE SAME LOGIC FOR HYBRIS TO ONE BAND DATASETS
        # Interpolate missing values using linear interpolation
        s1["daily_index"] = (s1[s1_band]
            .interpolate(method="linear")  # Fill missing values
        )

        s1['daily_index_smooth'] = s1["daily_index"].rolling(window=30, center=True, min_periods=1).mean()  # Apply rolling average
        
        #Find maxima and minima in the fused time series
        maxima = find_maxima(s1) #Peaks of seasons
        minima = find_minima(s1, prominenceMin = 0.1, distanceTillages=30, prominenceTillages=(0,1)) #Sowing, harvest, tillage

        #Identify growing seasons based on predicted minima
        g_seasons = growing_seasons(maxima, minima) #assign sowing and harvest dates to a peak of season

        #Add tillage predictions
        predictions = add_tillages(g_seasons, minima)

        #Check if field exists
        if field.empty:
            print(f"Field ID {id} not found in ground truth data.")

        #If field exists, proceed with analysis    
        else:

            #Merge with predictions
            s1 = add_predictions(s1, predictions)
            
            # Convert Date columns to datetime format
            field.loc[:, 'Date'] = pd.to_datetime(field['Date'], errors='coerce')

            s1 = merge_with_GT(s1, field)
            s1["ID"] = int(s1['ID'].unique()[0])

            results = validate_predictions(s1)

            return results, s1


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
    
    distanceMax : int, optional (default=60)
        Minimum distance between detected maxima.
    
    prominenceMax : tuple (default=(0.1, 1))
        Minimum prominence required for maxima.

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

def find_minima(hybris, distanceMin=15, prominenceMin=(0.1, 1), distanceTillages=30, prominenceTillages=(0.1, 1), height = None, wlen = None):
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

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing minima detected in the time series, with columns:
        - `Date`
        - `Value`
        - `Prominence`
    """
    #to find minima, a negative sign is put on the time series, as the find_peaks function only finds maxima.
    #by flipping the time series, minima become maxima.

    ### SOWING
    # Detect all smooth minima in the time series (sowing)
    sowing, sow_properties = find_peaks(-hybris["daily_index_smooth"], distance=distanceMin, prominence=prominenceMin, wlen = wlen)
    sow_properties = sow_properties["prominences"]


    ### HARVEST ORIGINAL
    # Detect all minima in the time series in rough time series (harvest)
    harvest, harv_properties = find_peaks(-hybris["daily_index"], distance=distanceMin, prominence=prominenceMin, wlen = wlen)
    harv_properties = harv_properties["prominences"]

    ### TILLAGE
    #Detect minima in the time series in rough time series (tillage)
    tillage, tillage_properties = find_peaks(-hybris["daily_index"], distance=distanceTillages, prominence=prominenceTillages, height=height, wlen = wlen)
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
            #Original way, take the first harvest
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
    
    # Build long dataframe
    df_long = pd.DataFrame(results).sort_values(["season_id", "Date"])
    
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

    # ensure event order on identical dates:
    # - pred_harvest before pred_tillage, so a tillage on harvest date belongs to the next season
    # - pred_tillage before pred_sowing, so a tillage on sowing date belongs to the previous season
    event_order = {
        "pred_harvest": 0,
        "pred_tillage": 1,
        "pred_sowing": 2
    }
    
    # Merge into hybris
    results = (
        pd.concat([g_seasons, tillage_df], ignore_index=True)
                .assign(event_rank=lambda df: df["pred_type"].map(event_order).fillna(10))
                .sort_values(["Date", "event_rank"])
                .drop(columns="event_rank")
                )
    return results

def add_predictions(hybris, predictions):

    hybris = hybris.copy().rename(columns={"date": "Date"})
    # Merge hybris time series with predicted events
    predictions = pd.merge(hybris, predictions, on="Date", how="left")

    # Add a flag True or False to assign to each date: is it part of a growing season or not?
    event_order = {
        "pred_harvest": 0,
        "pred_tillage": 1,
        "pred_sowing": 2,
        "peak": 3,
        "pred_tillage_excluded": 1
    }
    predictions = (
        predictions
        .assign(event_rank=lambda df: df["pred_type"].map(event_order).fillna(10))
        .sort_values(["Date", "event_rank"]).reset_index(drop=True)
        .drop(columns="event_rank")
    )

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


    # assign season number to each row, breaking also when a new sowing follows a harvest
    predictions['season_number'] = (
        ((predictions['in_growing_season'] != predictions['in_growing_season'].shift()) |
         ((predictions['pred_type'] == 'pred_sowing') & (predictions['pred_type'].shift() == 'pred_harvest')))
        .cumsum()
    )

    #Coincident tillages are kept! But filter out the tillage predicions which are inside a growing season
    mask = (
        (predictions["pred_type"] == "pred_tillage") &
        (predictions["in_growing_season"] == True) &
        (predictions["coincident"] == False)
    )

    predictions.loc[mask, "pred_type"] = 'pred_tillage_excluded'

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


def _ensure_datetime_and_sort(df, date_col="Date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ "ID", date_col ]).reset_index(drop=False)  # keep original index in 'index' column
    # keep original index as column so we can refer back to original rows
    df = df.rename(columns={"index": "_orig_index"})
    return df

def _empty_pred_dict():
    return {
        "pred_date": pd.NA,
        "pred_value": pd.NA,
        "pred_type": pd.NA,
        "pred_prominence": pd.NA,
        "pred_note": pd.NA,
        "delta": pd.NA,
        "pred_s1_contribution": pd.NA,
        "pred_s2_contribution": pd.NA,
        "matched": False
    }

def _build_comparison_dict(obs_row, pred_info, obs_type_label):
    d = {
        "ID": obs_row["ID"],
        "Farm": obs_row.get("Farm", pd.NA),
        "Country": obs_row.get("Country", pd.NA),
        "Crop": obs_row.get("Crop", pd.NA),
        "Crop.type": obs_row.get("Crop.type", pd.NA),
        "Method_EN": obs_row.get("Method_EN", pd.NA),
        "obs_date": obs_row["Date"],
        "obs_type": obs_type_label,
    }
    d.update(pred_info)
    return d

def _choose_peak_for_obs(peaks_df, obs_date, direction="future"):
    """
    peaks_df must be sorted by Date ascending.
    direction: "future" -> earliest peak strictly after obs_date
            "past"   -> latest peak strictly before obs_date
    Returns the peak row (as Series) or None.
    """
    if peaks_df.empty:
        return None
    if direction == "future":
        candidates = peaks_df[peaks_df["Date"] > obs_date]
        if candidates.empty:
            return None
        return candidates.iloc[0]  # earliest after obs_date
    else:
        # allow peaks up to obs_date
        cutoff = obs_date
        candidates = peaks_df[peaks_df["Date"] < cutoff]
        if candidates.empty:
            return None
        return candidates.iloc[-1]  # latest before cutoff

def _select_pred_within_season(merged_df, season_id, ID, pred_type_label, obs_date):
    """
    From merged_df select rows with given season_id, ID and pred_type_label.
    If multiple rows exist, choose the one closest in absolute days to obs_date.
    Returns the selected row (Series) or None.
    """
    preds = merged_df[
        (merged_df["season_id"] == season_id) &
        (merged_df["ID"] == ID) &
        (merged_df["pred_type"] == pred_type_label)
    ]
    if preds.empty:
        return None
    preds = preds.copy()
    preds["delta_abs"] = (preds["Date"] - obs_date).abs().dt.days
    chosen = preds.sort_values("delta_abs").iloc[0]
    return chosen

def _match_sowings_and_harvests(merged, peaks, obs_rows, direction, obs_type_label, pred_type_label):
    comparisons = []
    # ensure peaks sorted by Date per ID
    peaks_sorted = peaks.sort_values(["ID", "Date"])
    for _, obs_row in obs_rows.iterrows():
        # filter peaks for this ID
        peaks_for_id = peaks_sorted[peaks_sorted["ID"] == obs_row["ID"]]
        peak_row = _choose_peak_for_obs(peaks_for_id, obs_row["Date"], direction=direction)
        if peak_row is None:
            pred_info = _empty_pred_dict()
        else:
            season_id = peak_row["season_id"]
            pred_row = _select_pred_within_season(merged, season_id, obs_row["ID"], pred_type_label, obs_row["Date"])
            if pred_row is None:
                pred_info = _empty_pred_dict()
            else:
                delta_days = (pred_row["Date"] - obs_row["Date"]).days
                pred_info = {
                    "pred_date": pred_row["Date"],
                    "pred_value": pred_row.get("daily_index_smooth", pred_row.get("daily_index", pd.NA)),
                    "pred_type": pred_type_label,
                    "pred_prominence": pred_row.get("Prominence", pd.NA),
                    "pred_note": pred_row.get("season_complete", pd.NA),
                    "delta": delta_days,
                    "pred_s1_contribution": pred_row.get("s1_contribution", pd.NA),
                    "pred_s2_contribution": pred_row.get("s2_contribution", pd.NA),
                    "matched": True
                }
        comparisons.append(_build_comparison_dict(obs_row, pred_info, obs_type_label))
    return comparisons

def _match_tillages(merged, pred_tillages, coincidents, obs_tillages, one_to_one_tillage_match=True, add_window_for_tillages=True):
    """
    Returns list of comparison dicts for tillages and also returns unmatched prediction rows as comparison dicts
    (same structure as original function).
    - pred_tillages: DataFrame of predicted tillages (must include original index column '_orig_index')
    - coincidents: DataFrame of coincident predicted tillages (same columns)
    - obs_tillages: DataFrame of observed tillages (must include 'in_growing_season' boolean and 'season_number')
    """
    comparisons = []
    matched_pred_orig_indices = set()
    all_preds_in_window = []

    # Split observed tillages
    obs_within_dormant = obs_tillages[obs_tillages['in_growing_season'] == False]
    obs_in_growing = obs_tillages[obs_tillages['in_growing_season'] == True]

    # Precompute pred_tillages with original indices preserved
    pred_tillages = pred_tillages.copy()
    if "_orig_index" not in pred_tillages.columns:
        pred_tillages["_orig_index"] = pred_tillages.index

    coincidents = coincidents.copy()
    if "_orig_index" not in coincidents.columns:
        coincidents["_orig_index"] = coincidents.index

    # Dormant-season matching: match observed tillage to predictions in same dormant season_number
    for _, obs_row in obs_within_dormant.iterrows():
        obs_date = obs_row["Date"]
        obs_season = obs_row["season_number"]

        preds_in_window = pred_tillages[pred_tillages["season_number"] == obs_season].copy()
        # optionally remove already matched predictions using original indices
        if one_to_one_tillage_match:
            preds_in_window = preds_in_window[~preds_in_window["_orig_index"].isin(matched_pred_orig_indices)]

        all_preds_in_window.append(preds_in_window)

        # allow coincident tillages from current season
        coincident_candidates = coincidents[coincidents["season_number"].isin([obs_season])].copy()
        if one_to_one_tillage_match:
            coincident_candidates = coincident_candidates[~coincident_candidates["_orig_index"].isin(matched_pred_orig_indices)]

        if preds_in_window.empty and coincident_candidates.empty:
            comparisons.append(_build_comparison_dict(obs_row, _empty_pred_dict(), "obs_tillage"))
            continue

        if preds_in_window.empty:
            window_and_coincident = coincident_candidates
        else:
            window_and_coincident = pd.concat([preds_in_window, coincident_candidates], ignore_index=False).drop_duplicates(subset=["_orig_index"])

        window_and_coincident = window_and_coincident.copy()
        window_and_coincident["delta"] = (window_and_coincident["Date"] - obs_date).dt.days
        window_and_coincident["abs_delta"] = window_and_coincident["delta"].abs()
        # choose the row with smallest abs_delta; keep its original index
        chosen = window_and_coincident.sort_values("abs_delta").iloc[0]
        chosen_orig_index = chosen["_orig_index"]
        delta = chosen["delta"]

        pred_info = {
            "pred_date": chosen["Date"],
            "pred_value": chosen.get("daily_index", pd.NA),
            "pred_type": "pred_tillage",
            "pred_prominence": chosen.get("Prominence", pd.NA),
            "pred_note": "in_dormant_season",
            "delta": int(delta) if pd.notna(delta) else pd.NA,
            "pred_s1_contribution": chosen.get("s1_contribution", pd.NA),
            "pred_s2_contribution": chosen.get("s2_contribution", pd.NA),
            "matched": True
        }
        comparisons.append(_build_comparison_dict(obs_row, pred_info, "obs_tillage"))
        matched_pred_orig_indices.add(chosen_orig_index)

    # Growing-season tillages: match to closest predicted tillage across all predictions (optionally with 30-day window)
    for _, obs_row in obs_in_growing.iterrows():
        obs_date = obs_row["Date"]
        preds_all = pred_tillages.copy()
        preds_all["delta"] = (preds_all["Date"] - obs_date).dt.days
        preds_all["abs_delta"] = preds_all["delta"].abs()
        preds_all = preds_all.sort_values("abs_delta")
        if preds_all.empty:
            comparisons.append(_build_comparison_dict(obs_row, _empty_pred_dict(), "obs_tillage"))
            continue
        chosen = preds_all.iloc[0]
        delta = int(chosen["delta"])
        chosen_orig_index = chosen["_orig_index"]

        if add_window_for_tillages and abs(delta) > 30:
            comparisons.append(_build_comparison_dict(obs_row, _empty_pred_dict(), "obs_tillage"))
            continue

        pred_info = {
            "pred_date": chosen["Date"],
            "pred_value": chosen.get("daily_index", pd.NA),
            "pred_type": "pred_tillage",
            "pred_prominence": chosen.get("Prominence", pd.NA),
            "pred_note": "in_growing_season",
            "delta": delta,
            "pred_s1_contribution": chosen.get("s1_contribution", pd.NA),
            "pred_s2_contribution": chosen.get("s2_contribution", pd.NA),
            "matched": True
        }
        comparisons.append(_build_comparison_dict(obs_row, pred_info, "obs_tillage"))
        if one_to_one_tillage_match:
            matched_pred_orig_indices.add(chosen_orig_index)

    # Add unmatched predictions from all_preds_in_window (use original indices)
    if all_preds_in_window:
        all_preds_in_window_df = pd.concat(all_preds_in_window, ignore_index=False).drop_duplicates(subset=["_orig_index"])
        # select preds whose original index is not in matched_pred_orig_indices
        unmatched_preds = all_preds_in_window_df[~all_preds_in_window_df["_orig_index"].isin(matched_pred_orig_indices)]
        for _, row in unmatched_preds.iterrows():
            pred_info = {
                "pred_date": row["Date"],
                "pred_value": row.get("daily_index", pd.NA),
                "pred_type": "pred_tillage",
                "pred_prominence": row.get("Prominence", pd.NA),
                "pred_note": "in_dormant_season",
                "delta": pd.NA,
                "pred_s1_contribution": row.get("s1_contribution", pd.NA),
                "pred_s2_contribution": row.get("s2_contribution", pd.NA),
                "matched": False
            }
            d = {
                "ID": row.get("ID", pd.NA),
                "Farm": row.get("Farm", pd.NA),
                "Country": row.get("Country", pd.NA),
                "Crop": row.get("Crop", pd.NA),
                "Crop.type": row.get("Crop.type", pd.NA),
                "Method_EN": row.get("Method_EN", pd.NA),
                "obs_date": pd.NA,
                "obs_type": pd.NA,
            }
            d.update(pred_info)
            comparisons.append(d)

    return comparisons

def validate_predictions(merged, one_to_one_tillage_match=True, add_window_for_tillages=False):
    """
    Replacement for validate_predictions.
    - merged: input DataFrame with Date, pred_type, obs_type, season_id, season_number, ID, etc.
    - one_to_one_tillage_match: if True, ensure a predicted tillage is matched at most once
    - add_window_for_tillages: if True, apply 30-day window rule for tillages in growing season
    Returns: comparison_df with same columns as original function output.
    """
    # Prepare data and keep original indices
    merged_prepared = _ensure_datetime_and_sort(merged, date_col="Date")

    # Recreate the filtered sets
    peaks = merged[merged["pred_type"] == "peak"].sort_values(["ID", "Date"])
    obs_sowings = merged[merged["obs_type"] == "obs_sowing"].drop_duplicates(subset=["Date", "ID"])  # ensure unique observed sowings by date and ID
    obs_harvests = merged[merged["obs_type"] == "obs_harvest"].drop_duplicates(subset=["Date", "ID"])  # ensure unique observed harvests by date and ID
    obs_tillages = merged[merged["obs_type"] == "obs_tillage"].drop_duplicates(subset=["Date", "ID"])  # ensure unique observed tillages by date and ID
    pred_tillages = merged[merged["pred_type"] == "pred_tillage"].drop_duplicates(subset=["Date", "ID"])  # ensure unique predicted tillages by date and ID
    coincidents = merged[(merged["pred_type"] == "pred_tillage") & (merged.get("coincident", False) == True)]

    comparisons = []

    # Sowings: observed sowing -> earliest future peak -> pred_sowing in that season
    comparisons += _match_sowings_and_harvests(merged_prepared, peaks, obs_sowings, direction="future",
                                            obs_type_label="obs_sowing", pred_type_label="pred_sowing")

    # Harvests: observed harvest -> latest past peak -> pred_harvest in that season
    comparisons += _match_sowings_and_harvests(merged_prepared, peaks, obs_harvests, direction="past",
                                            obs_type_label="obs_harvest", pred_type_label="pred_harvest")

    # Tillages
    tillage_comparisons = _match_tillages(merged_prepared, pred_tillages, coincidents, obs_tillages,
                                        one_to_one_tillage_match=one_to_one_tillage_match,
                                        add_window_for_tillages=add_window_for_tillages)
    comparisons += tillage_comparisons

    comparison_df = pd.DataFrame(comparisons).drop_duplicates().reset_index(drop=True)
    return comparison_df

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

def plot_hybris(hybris, plot_tillages = True, plot_dormant=False, add_groundtruth = True, add_pred_sow_harv=True, ax = None, date_col = 'date', color = 'brown', legend = True, alpha_smooth = 1, alpha_raw = 0.3):
    
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

    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5)) # (x, y) coordinates relative to the axes)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)
    if show:
        plt.show()

    return ax