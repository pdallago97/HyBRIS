#HyBRIS playground
from HyBRIS_utils import *
import pickle

#load data
save_dictionary = False # set to True to compute the dictionary with all combinations of observations (this can take a while), set to False to load the precomputed dictionary


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
start_date = pd.to_datetime("2022-01-01")
end_date = pd.to_datetime("2023-01-01")

print(hybris.head())
print(hybris.columns.tolist())

#Filter all data to the same time period for easier plotting
field_example = filter_by_date(field, start_date=start_date, end_date=end_date, date_column="Date")
s2_example = filter_by_date(s2, start_date=start_date, end_date=end_date)
s1_example = filter_by_date(s1, start_date=start_date, end_date=end_date)
pp_example = filter_by_date(pp, start_date=start_date, end_date=end_date)

# define function to randomly mask observations from the Sentinel 1 or Sentinel 2 time series
# to explore the effect of data availability on the HyBRIS index 
def limit_observations(df, n_obs, date_col="date"):
    """
    Randomly keep only n observations per year
    to simulate availability.
    """
    if n_obs is None:
        return df

    df = df.copy()
    df["year"] = df[date_col].dt.year

    out = (
        df.groupby("year", group_keys=False)
        .apply(
              lambda x: x.sample(min(len(x), n_obs)),include_groups=False
          )
        .sort_values(date_col)
    )

    return out

from functools import lru_cache

@lru_cache(maxsize=None)
def cached_hybris(s1_obs, s2_obs):
    s1_sub = limit_observations(s1_example, s1_obs)
    s2_sub = limit_observations(s2_example, s2_obs)
    return calculate_hybris(s1_sub, s2_sub), s1_sub, s2_sub

if save_dictionary: # set to True to compute the dictionary with all combinations of observations (this can take a while)
    precomputed = {}
    for s1_obs in range(10, len(s1_example)+1, 2):
        for s2_obs in range(5, len(s2_example)+1, 2):
            precomputed[(s1_obs, s2_obs)] = cached_hybris(s1_obs, s2_obs)

    print(precomputed.keys())

    with open(r'C:\MyData\Python\HyBRIS\Experiments\variations_in_S1_S2_variability.pickle', 'wb') as handle:
        pickle.dump(precomputed, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    #load dictionary
    with open(r'C:\MyData\Python\HyBRIS\Experiments\variations_in_S1_S2_variability.pickle', 'rb') as handle:
        precomputed = pickle.load(handle)

    print(precomputed.keys())

####PLOT FIGURE WITH 1 PANELS
#invert hybris for plotting
# weighted_mean = hybris
# weighted_mean.loc[:, 'daily_index'] = 1 - hybris['daily_index']
# weighted_mean.loc[:, 'daily_index_smooth'] = 1 - hybris['daily_index_smooth']
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

max_s1 = len(s1_example)
max_s2 = len(s2_example)

cmap_s1 = cm.Blues      # radar
cmap_s2 = cm.OrRd       # optical
cmap_hybris = cm.viridis
norm_s1 = mcolors.Normalize(vmin=1, vmax=max_s1)
norm_s2 = mcolors.Normalize(vmin=1, vmax=max_s2)
norm_h = mcolors.Normalize(vmin=2, vmax=max_s1 + max_s2)
sm_s1 = ScalarMappable(norm=norm_s1, cmap=cmap_s1)
sm_s2 = ScalarMappable(norm=norm_s2, cmap=cmap_s2)
sm_h  = ScalarMappable(norm=norm_h, cmap=cmap_hybris)

all_maxima = []
all_minima = []

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11.5, 4.1), sharex=True)
for (s1_obs, s2_obs), (hybris_sub, s1_sub, s2_sub) in precomputed.items():

    # ---------- colors ----------
    color_s1 = cmap_s1(norm_s1(s1_obs))
    color_s2 = cmap_s2(norm_s2(s2_obs))
    color_h  = cmap_hybris(norm_h(s1_obs + s2_obs))
    if s1_obs == max_s1 and s2_obs == max_s2: # if we have all observations, use black for better visibility
        color_h = 'black'
        color_s2 = 'black'
        color_s1 = 'black'

    plot_hybris(hybris_sub[(hybris_sub["date"] > start_date) & (hybris_sub["date"] < end_date)],
                add_pred_sow_harv = False,
                plot_tillages=False,
                plot_dormant=False,
                add_groundtruth=False,
                ax = axes[0], date_col = "date",
                color = color_h, # color based on number of observations
                legend = False
                )
    #add BSI and VV/VH time series to the plot
    plot_time_series(s2_sub[s2_band], s2_sub['date'], color=color_s2, title=s2_band + f" (n={len(s2_sub)})", add=True,
                    show=False, alpha=0.8, marker='o', ax=axes[1], legend = False)
    plot_time_series(s1_sub[s1_band], s1_sub['date'], color=color_s1, title="VV/VH" + f" (n={len(s1_sub)})",
                    add=True, show=False, alpha=0.8, marker='o', ax=axes[2], legend=False)

fig.colorbar(sm_h, ax=axes[0], label="Total observations (S1 + S2)")
fig.colorbar(sm_s2, ax=axes[1], label="Sentinel-2 observations")
fig.colorbar(sm_s1, ax=axes[2], label="Sentinel-1 observations")
axes[0].set_title("HyBRIS response to data availability")
axes[1].set_title("Optical signal (BSI)")
axes[2].set_title("Radar signal (VV/VH)")
plt.tight_layout()
plt.show()