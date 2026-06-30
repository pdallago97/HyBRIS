from HyBRIS_utils import *

#specify bands for S1 and S2 fusion
#Here for example we merge NDVI and VH/VV
s2_band = "NDVI"
s1_band = "VH_VV"

#paths to example data files, the Sentinel 1 and 2 time series
s2_path = 'Example/Sentinel2_example.csv'
s1_path = 'Example/Sentinel1_example.csv'

# Define bands of the Sentinel 1 and 2 files (make sure the band names match the columns of the .csv time series)
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

## CREATE HYBRID INDEX
#Our Sentinel 1 and 2 time series are ready to be merged: use daily_index_with_contributions_vectorized() to merge any bands
hybrid_vegetation_index = daily_index_with_contributions_vectorized(s2,s1, bandS2=s2_band, bandS1=s1_band)
print(hybrid_vegetation_index.head())


####PLOT FIGURE WITH 3 PANELS
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.2, 11.5), sharex=True)

# --- First panel: Sentinel-2
plot_time_series(s2[s2_band], s2['date'], color='green', title=s2_band, legend = True,
                 show=False, alpha=0.8, marker='o', ax=axes[0])
axes[0].set_title(s2_band + " - Sentinel 2", fontsize=14)

# --- Second panel: Sentinel-1
plot_time_series(s1[s1_band], s1['date'], color='blue', title=s1_band, legend = True,
                 add=False, show=False, alpha=0.8, marker='o', ax=axes[1])
axes[1].set_title(s1_band + " - Sentinel 1", fontsize=14)

# --- Third panel: Hybrid index
plot_time_series(hybrid_vegetation_index["daily_index"], hybrid_vegetation_index['date'], color='purple', title="hybrid index smoothed", legend = True,
                 add=False, show=False, ax=axes[2], alpha= 0.3)
plot_time_series(hybrid_vegetation_index["daily_index_smooth"], hybrid_vegetation_index['date'], color='purple', title="hybrid index unsmoothed", legend = True,
                 add=True, show=False, ax=axes[2], alpha=1)
axes[2].set_title("Hybrid vegetation index", fontsize=14)

plt.tight_layout()
plt.show()
