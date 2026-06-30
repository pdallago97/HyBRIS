from HyBRIS_utils import *

#Ground truth data with sowing and harvest dates
field = pd.read_csv('Example/GroundTruth_example.csv')
field['Date'] = pd.to_datetime(field['Date']) #make sure the date column is in datetime format

#paths to example data files, the Sentinel 1 and 2 time series
s2_path = 'Example/Sentinel2_example.csv'
s1_path = 'Example/Sentinel1_example.csv'

#define start and end date for examples (indices will be plotted in this range)
start_date = pd.to_datetime("2019-01-01")
end_date = pd.to_datetime("2021-08-01")

# Define bands of the Sentinel 1 and 2 files
bandsused = ['B1', 'B2', 'B3', 'B4','B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
bandsusedS1 = ['VV', 'VH', 'angle']

### PROCESS SENTINEL-2 (Optical)
# Open Sentinel-2, remove NA, merge observation on the same day, and add Vegetation indices
s2 = openSentinel2file(s2_path, bandsused)
s2 = add_vis(s2)

## PROCESS SENTINEL-1 (Radar)
# Open Sentinel-1 files, remove NA, merge observation on the same day, select most present orbit, and add Vegetation indices
s1 = openSentinel1file(s1_path, bandsusedS1)

#Split ascending and descending orbits
s1_des = s1[s1['orbit'] == 'DESCENDING']  # Select only descending orbits
s1_asc = s1[s1['orbit'] == 'ASCENDING']  # Select only ascending orbits

# Select most present orbit and add radar vegetation index
s1_des = add_vis_radar(selectOrbit(s1_des, selectMostPresent=True))
s1_asc = add_vis_radar(selectOrbit(s1_asc, selectMostPresent=True))

s1 = pd.concat([s1_des, s1_asc])

####################### DETECT FARMING PRACTICES WITH HYBRID DATA ###################################
#specify bands for S1 and S2 fusion
s2_band = "BSI"
s1_band = "VH_VV"

#The daily_index_with_contributions_vectorized() function is used also for HyBRIS, but has been wrapped in the following function:
hybris = calculate_hybris_vectorized(s1, s2)

#Find maxima and minima in the fused time series
maxima = find_maxima(hybris) #Peaks of seasons
minima = find_minima(hybris, distanceTillages=30, prominenceTillages=(0,1), height= None) #Sowing, harvest, tillage

#Identify growing seasons based on predicted minima
g_seasons = growing_seasons(maxima, minima) #assign sowing and harvest dates to a peak of season

#Add tillages to predictions
predictions = add_tillages(g_seasons, minima)

#Merge with predictions
hybris = add_predictions(hybris, predictions)

#Merge with ground truth data
hybris = merge_with_GT(hybris, field) #combine everything in one long dataframe

# The objext hybris, now is a long time series which contains all the ground truth observations, and their predictions
print("HyBRIS data, field data, and predictions are stored in a long dataframe, sorted by Date: \n")
print(hybris.head())

####################### INTERPOLATE OPTICAL DATA ###################################
# Invert BSI index
if s2_band == 'BSI':
   s2[s2_band] = 1 - s2[s2_band] # Invert the index for BSI

# Create a daily date range spanning the entire dataset
daily_dates = pd.DataFrame({"date": pd.date_range(start=s2["date"].min(), end=s2["date"].max(), freq="D")})

# Merge with an outer join to keep all observations
s2 = pd.merge(s2, daily_dates, on="date", how="outer")

# Interpolate missing values using linear interpolation
s2["daily_index"] = s2[s2_band].interpolate(method="linear")  # Fill missing values

s2["daily_index_smooth"] = s2["daily_index"].rolling(window=30, center=True, min_periods=1).mean()  # Apply rolling average

####################### DETECT WITH OPTICAL DATA ###################################
#Find maxima and minima in the fused time series
maxima_s2 = find_maxima(s2) #Peaks of seasons
minima_s2 = find_minima(s2, distanceTillages=30, prominenceTillages=(0,1), height= None) #Sowing, harvest, tillage

#Identify growing seasons based on predicted minima
g_seasons_s2 = growing_seasons(maxima_s2, minima_s2) #assign sowing and harvest dates to a peak of season

predictions_s2 = add_tillages(g_seasons_s2, minima_s2)

#Check if field exists
if field.empty:
    print(f"Field ID {id} not found in ground truth data.")

#If field exists, proceed with analysis    
else:
    #Merge with predictions
    s2 = add_predictions(s2, predictions_s2)
    
    # Convert Date columns to datetime format
    field.loc[:, 'Date'] = pd.to_datetime(field['Date'], errors='coerce')

    s2 = merge_with_GT(s2, field)
    s2["ID"] = int(s2['ID'].unique()[0])

# The objext s2, now is a long time series which contains all the ground truth observations, and their predictions
print("Sentinel-2 data, field data, and predictions are stored in a long dataframe, sorted by Date: \n")
print(s2.head())

####################### INTERPOLATE RADAR DATA ###################################
# Create a daily date range spanning the entire dataset
daily_dates = pd.DataFrame({"date": pd.date_range(start=s1["date"].min(), end=s1["date"].max(), freq="D")})

# Merge with an outer join to keep all observations
s1 = pd.merge(s1, daily_dates, on="date", how="outer")

# Interpolate missing values using linear interpolation
s1["daily_index"] = s1[s1_band].interpolate(method="linear")  # Fill missing values

s1["daily_index_smooth"] = s1["daily_index"].rolling(window=30, center=True, min_periods=1).mean()  # Apply rolling average

####################### DETECT WITH RADAR DATA ###################################
#Find maxima and minima in the fused time series
maxima_s1 = find_maxima(s1) #Peaks of seasons
minima_s1 = find_minima(s1, distanceTillages=30, prominenceTillages=(0,1), height= None) #Sowing, harvest, tillage

#Identify growing seasons based on predicted minima
g_seasons_s1 = growing_seasons(maxima_s1, minima_s1) #assign sowing and harvest dates to a peak of season

predictions_s1 = add_tillages(g_seasons_s1, minima_s1)

#Check if field exists
if field.empty:
    print(f"Field ID {id} not found in ground truth data.")

#If field exists, proceed with analysis    
else:
    #Merge with predictions
    s1 = add_predictions(s1, predictions_s1)
    
    # Convert Date columns to datetime format
    field.loc[:, 'Date'] = pd.to_datetime(field['Date'], errors='coerce')

    s1 = merge_with_GT(s1, field)
    s1["ID"] = int(s1['ID'].unique()[0])

# The objext s1, now is a long time series which contains all the ground truth observations, and their predictions
print("Sentinel-1 data, field data, and predictions are stored in a long dataframe, sorted by Date: \n")
print(s1.head())

####################### PLOT RESULTS ###################################
#Filter all data to the same time period for easier plotting
field_example = filter_by_date(field, start_date=start_date, end_date=end_date, date_column="Date")
s2_example = filter_by_date(s2, start_date=start_date, end_date=end_date, date_column="Date")
s1_example = filter_by_date(s1, start_date=start_date, end_date=end_date, date_column="Date")
minima_example = filter_by_date(minima, start_date=start_date, end_date=end_date, date_column="Date")

####PLOT FIGURE WITH 3 PANELS
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.2, 11.5), sharex=True)

# --- First panel: Sentinel-2
plot_hybris(s2_example, ax = axes[0], date_col = 'Date', color = "green", plot_tillages = False, add_pred_sow_harv = True)
plot_time_series(s2_example["BSI"], s2_example['Date'], color='green', title="", legend = False,
                 show=False, alpha=0.2, marker='o', ax=axes[0])

axes[0].set_title("Inverted BSI - Sentinel 2", fontsize=14)

# --- Second panel: Sentinel-1
plot_hybris(s1_example, ax = axes[1], date_col = 'Date', color = "blue", plot_tillages = False, add_pred_sow_harv = True)
plot_time_series(s1_example["VH_VV"], s1_example['Date'], color='blue', title="", legend = False,
                 add=False, show=False, alpha=0.2, marker='o', ax=axes[1])

axes[1].set_title("VH/VV - Sentinel 1", fontsize=14)

# --- Third panel: plot_hybris with groundtruth and predictions
plot_hybris(hybris[(hybris["Date"] > start_date) & (hybris["Date"] < end_date)], add_pred_sow_harv = True,
            plot_tillages=False, plot_dormant=False, add_groundtruth=True, ax=axes[2], date_col="Date")

axes[2].set_title("HyBRIS - Sentinel1&2", fontsize=14)

# Optional: tighten layout
plt.tight_layout()
plt.show()

####PLOT FIGURE WITH 1 PANEL
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.5, 4.1), sharex=True)
plot_hybris(hybris[(hybris["Date"] > start_date) & (hybris["Date"] < end_date)],
            add_pred_sow_harv = True,
            plot_tillages=True,
            plot_dormant=True,
            add_groundtruth=True,
            ax = axes, date_col = "Date")
axes.set_title("HyBRIS - Sentinel1&2 with tillages and dormant periods", fontsize=14)
plt.tight_layout()
plt.show()

