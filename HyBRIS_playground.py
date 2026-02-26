#HyBRIS playground
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

print("--- Sentinel-2 data ---")
print(s2.head())
print(s2.columns.tolist())
print("length of Sentinel-2 data:", len(s2))

#and Sentinel-1 files, remove NA, merge observation on the same day, select most present orbit, and add Vegetation indices
s1 = openSentinel1file(s1_path, bandsusedS1)

#Split ascending and descending orbits
s1_des = s1[s1['orbit'] == 'DESCENDING']  # Select only descending orbits
s1_asc = s1[s1['orbit'] == 'ASCENDING']  # Select only ascending orbits

# Select most present orbit and add radar vegetation index
s1_des = add_vis_radar(selectOrbit(s1_des, selectMostPresent=True))
s1_asc = add_vis_radar(selectOrbit(s1_asc, selectMostPresent=True))

s1 = pd.concat([s1_des, s1_asc])

print("--- Sentinel-1 data ---")
print(s1.head())
print(s1.columns.tolist())
print("length of Sentinel-1 data:", len(s1))

#specify bands for S1 and S2 fusion
s2_band = "BSI"
s1_band = "VV_VH"

#Fuse S1 and S2 time series based on given bands
pp = daily_index_with_contributions(s2,s1, maxDiff= 30, bandS2= "BSI", bandS1= "VV_VH")
pp['daily_index'] = 1 - pp['daily_index']

#Fuse S1 and S2 time series based on given bands(similar to daily_index_with_contributions, but specifically for HyBRIS)
hybris = calculate_hybris(s1, s2)
print("--- HyBRIS data ---")
print(hybris.head())
print(hybris.columns.tolist())
print("length of HyBRIS data:", len(hybris))

#define start and end date for examples
start_date = pd.to_datetime("2017-01-01")
end_date = pd.to_datetime("2025-01-01")

#Filter all data to the same time period for easier plotting
field_example = filter_by_date(field, start_date=start_date, end_date=end_date, date_column="Date")
s2_example = filter_by_date(s2, start_date=start_date, end_date=end_date)
s1_example = filter_by_date(s1, start_date=start_date, end_date=end_date)

####PLOT FIGURE WITH 1 PANELS
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.5, 4.1), sharex=True)
plot_hybris(hybris[(hybris["date"] > start_date) & (hybris["date"] < end_date)],
            add_pred_sow_harv = False,
            plot_tillages=False,
            plot_dormant=False,
            add_groundtruth=False,
            ax = axes, date_col = "date")

plt.tight_layout()
plt.show()

####PLOT FIGURE WITH 1 PANELS
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.5, 4.1), sharex=True)
plot_time_series(hybris["s1_contribution"], hybris['date'], color='blue', title="S1 contribution", show=False, alpha=0.5, marker='o', ax=axes)
plot_time_series(hybris["s2_contribution"], hybris['date'], color='green', title="S2 contribution", show=False, alpha=0.5, marker='o', ax=axes)

plt.tight_layout()
plt.show()
