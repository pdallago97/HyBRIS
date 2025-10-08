from HyBRIS_utils import *

#load data

#Ground truth data with sowing and harvest dates
field = pd.read_csv('Example/GroundTruth_example.csv')
field['Date'] = pd.to_datetime(field['Date']) #make sure the date column is in datetime format

#paths to example data files, the Sentinel 1 and 2 time series
s2_path = 'Example/Sentinel2_example.csv'
s1_path = 'Example/Sentinel1_example.csv'

# Open Sentinel-2, remove NA, merge observation on the same day, and add Vegetation indices
bandsused = ['B1', 'B2', 'B3', 'B4','B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
bandsusedS1 = ['VV', 'VH', 'angle']

s2 = openSentinel2file(s2_path, bandsused)
s2 = add_vis(s2)

#and Sentinel-1 files, remove NA, merge observation on the same day, select most present orbit, and add Vegetation indices
s1 = openSentinel1file(s1_path, bandsusedS1)

#Split ascending and descending orbits
s1_des = s1[s1['orbit'] == 'DESCENDING']  # Select only descending orbits
s1_asc = s1[s1['orbit'] == 'ASCENDING']  # Select only ascending orbits

# Select most present orbit and add radar vegetation index
s1_des = add_vis_radar(selectOrbit(s1_des, selectMostPresent=True))
s1_asc = add_vis_radar(selectOrbit(s1_asc, selectMostPresent=True))

s1 = pd.concat([s1_des, s1_asc])

#specify bands for S1 and S2 fusion
s2_band = "BSI"
s1_band = "VV_VH"

#Fuse S1 and S2 time series based on given bands
pp = daily_index_with_contributions(s2,s1, maxDiff= 30, bandS2= "BSI", bandS1= "VV_VH")
pp['daily_index'] = 1 - pp['daily_index']

#Fuse S1 and S2 time series based on given bands(similar to daily_index_with_contributions, but specifically for HyBRIS)
hybris = calculate_hybris(s1, s2)

#Find maxima and minima in the fused time series
maxima = find_maxima(hybris) #Peaks of seasons
minima = find_minima(hybris, distanceTillages=30, prominenceTillages=(0,1), height= None) #Sowing, harvest, tillage

#Identify growing seasons based on predicted minima
g_seasons = growing_seasons(maxima, minima) #assign sowing and harvest dates to a peak of season

#Filter out small seasons (less than 60 days)
g_seasons = filter_small_seasons(g_seasons)

predictions = add_tillages(g_seasons, minima)

#Merge with predictions
hybris = add_predictions(hybris, predictions)

#Merge with ground truth data
hybris = merge_with_GT(hybris, field) #combine everything in one long dataframe

#define start and end date for examples
start_date = pd.to_datetime("2017-01-01")
end_date = pd.to_datetime("2025-01-01")

#Filter all data to the same time period for easier plotting
field_example = filter_by_date(field, start_date=start_date, end_date=end_date, date_column="Date")
s2_example = filter_by_date(s2, start_date=start_date, end_date=end_date)
s1_example = filter_by_date(s1, start_date=start_date, end_date=end_date)
pp_example = filter_by_date(pp, start_date=start_date, end_date=end_date)
minima_example = filter_by_date(minima, start_date=start_date, end_date=end_date, date_column="Date")

####PLOT FIGURE WITH 1 PANELS
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.5, 4.1), sharex=True)
plot_hybris(hybris[(hybris["Date"] > start_date) & (hybris["Date"] < end_date)],
            add_pred_sow_harv = False,
            plot_tillages=False,
            plot_dormant=True,
            add_groundtruth=True,
            ax = axes)

plt.tight_layout()
plt.show()

####PLOT FIGURE WITH 4 PANELS
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8.2, 11.5), sharex=True)
# Invert Hybris index
pp_example.loc[:, 'daily_index'] = 1 - pp_example['daily_index']

# --- First panel: S2, S1, HYBRIS
plot_time_series(s2_example[s2_band], s2_example['date'], color='red', title=s2_band,
                 show=False, alpha=0.2, marker='o', ax=axes[0])
plot_time_series(s1_example[s1_band], s1_example['date'], color='purple', title="VV/VH",
                 add=True, show=False, alpha=0.2, marker='o', ax=axes[0])
plot_time_series(pp_example['daily_index'], pp_example['date'], color='brown', title="Xd",
                 add=True, ax=axes[0], show=False)

# Restore original HYBRIS index if needed
pp_example.loc[:, 'daily_index'] = 1 - pp_example['daily_index']

# --- Second panel: NDVI, VH/VV, HYBRIS
plot_time_series(s2_example["NDVI"], s2_example['date'], color='green', title="NDVI",
                 show=False, alpha=0.2, marker='o', ax=axes[1])
plot_time_series(s1_example["VH_VV"], s1_example['date'], color='blue', title="VH/VV",
                 add=True, show=False, alpha=0.2, marker='o', ax=axes[1])
plot_time_series(pp_example['daily_index'], pp_example['date'], color='brown',
                 title="HYBRIS unsmoothed", add=True, ax=axes[1], show=False)

# --- Third panel: plot_hybris without groundtruth
plot_hybris(hybris[(hybris["Date"] > start_date) & (hybris["Date"] < end_date)],
            plot_tillages=True, plot_dormant=True, add_groundtruth=False, ax=axes[2])

# --- Fourth panel: plot_hybris with groundtruth
plot_hybris(hybris[(hybris["Date"] > start_date) & (hybris["Date"] < end_date)], add_pred_sow_harv = False,
            plot_tillages=False, plot_dormant=False, add_groundtruth=True, ax=axes[3])

# Optional: tighten layout
plt.tight_layout()
plt.show()


