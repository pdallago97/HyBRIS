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
print("length of HyBRIS data:", len(hybris.index))


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


########### TRY TO USE SAME LOGIC FOR HYBRIS TO ONE BAND DATASETS
s2_band = "BSI"
# Create a daily date range spanning the entire dataset
daily_dates = pd.DataFrame({"date": pd.date_range(start=s2["date"].min(), end=s2["date"].max(), freq="D")})

# Merge with an outer join to keep all observations
s2 = pd.merge(s2, daily_dates, on="date", how="outer")

# Interpolate missing values using linear interpolation
s2["daily_index"] = (s2[s2_band]
    .interpolate(method="linear")  # Fill missing values
)

s2['daily_index_smooth'] = s2["daily_index"].rolling(window=30, center=True, min_periods=1).mean()  # Apply rolling average

print(s2["daily_index"], s2['date'])

if s2_band == "BSI":
    s2['daily_index'] = 1 - s2['daily_index']
    s2['daily_index_smooth'] = 1 - s2['daily_index_smooth']
    
#Find maxima and minima in the fused time series
maxima = find_maxima(s2) #Peaks of seasons
minima = find_minima(s2, distanceTillages=30, prominenceTillages=(0,1), height= None) #Sowing, harvest, tillage

#Identify growing seasons based on predicted minima
g_seasons = growing_seasons(maxima, minima) #assign sowing and harvest dates to a peak of season

#Filter out small seasons (less than 60 days)
g_seasons = filter_small_seasons(g_seasons)

predictions = add_tillages(g_seasons, minima)

#Merge with predictions
s2_example = add_predictions(s2_example, predictions)

#Merge with predictions
s2 = add_predictions(s2, predictions)

# Convert Date columns to datetime format
field.loc[:, 'Date'] = pd.to_datetime(field['Date'], errors='coerce')

s2 = merge_with_GT(s2, field)
s2["ID"] = 805 # add ID column for validation
s2.to_csv(r"C:\Users\dall002\OneDrive - Wageningen University & Research\Chapter 1\2nd_revision\ts_experiments\s2_exp.csv", index=False)
print(s2.head())
print(s2.columns.tolist())
results = validate_predictions(s2)
results.to_csv(r"C:\Users\dall002\OneDrive - Wageningen University & Research\Chapter 1\2nd_revision\ts_experiments\results_s2_exp.csv", index=False)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11.5, 4.1), sharex=True)
plot_hybris(s2,
            add_pred_sow_harv = True,
            plot_tillages=True,
            plot_dormant=True,
            add_groundtruth=True,
            ax = axes, date_col = "Date",
            color = 'blue', # color based on number of observations
            legend = True,
            alpha_raw=0.5, alpha_smooth=1
            )
print(results)
plt.tight_layout()
plt.show()

k+o

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
    s2_cache = filter_by_date(s2_df, start_date=start_date, end_date=end_date)
    s1_cache = filter_by_date(s1_df, start_date=start_date, end_date=end_date)

    #open file
    s1_sub = limit_observations(s1_cache, s1_obs)
    s2_sub = limit_observations(s2_cache, s2_obs)

    s2_sub = add_vis(s2_sub)
    #Split ascending and descending orbits
    s1_des_sub = s1_sub[s1_sub['orbit'] == 'DESCENDING']  # Select only descending orbits
    s1_asc_sub = s1_sub[s1_sub['orbit'] == 'ASCENDING']  # Select only ascending orbits

    # Select most present orbit and add radar vegetation index
    s1_des_cache = add_vis_radar(selectOrbit(s1_des_sub, selectMostPresent=True))
    s1_asc_cache = add_vis_radar(selectOrbit(s1_asc_sub, selectMostPresent=True))

    s1_sub = pd.concat([s1_des_cache, s1_asc_cache])

    return calculate_hybris(s1_sub, s2_sub), s1_sub, s2_sub

if save_dictionary: # set to True to compute the dictionary with all combinations of observations (this can take a while)
    precomputed = {}
    s1_df = openSentinel1file(s1_path, bandsusedS1)
    s2_df = openSentinel2file(s2_path, bandsused)

    for s1_obs in range(40, len(s1_example)+1, 2):
        for s2_obs in range(15, len(s2_example)+1, 2):
            precomputed[(s1_obs, s2_obs)] = cached_hybris(s1_obs, s2_obs)

    print(precomputed.keys())

    with open(r'C:\MyData\Python\HyBRIS\Experiments\variations_in_S1_S2_variability.pickle', 'wb') as handle:
        pickle.dump(precomputed, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    #load dictionary
    with open(r'C:\MyData\Python\HyBRIS\Experiments\variations_in_S1_S2_variability.pickle', 'rb') as handle:
        precomputed = pickle.load(handle)

    print(precomputed.keys())
    print("Total number of combinations:", len(precomputed.keys()))

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

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(11.5, 4.1), sharex=True)
for (s1_obs, s2_obs), (hybris_sub, s1_sub, s2_sub) in precomputed.items():
    add_sow_harv = False  # default to False
    try:
        #Find maxima and minima in the fused time series
        maxima = find_maxima(hybris_sub) #Peaks of seasons
        minima = find_minima(hybris_sub, distanceTillages=30, prominenceTillages=(0,1), height= None) #Sowing, harvest, tillage

        #Identify growing seasons based on predicted minima
        g_seasons = growing_seasons(maxima, minima) #assign sowing and harvest dates to a peak of season

        #Filter out small seasons (less than 60 days)
        g_seasons = filter_small_seasons(g_seasons)

        predictions = add_tillages(g_seasons, minima)

        #Merge with predictions
        hybris_sub = add_predictions(hybris_sub, predictions)
        
        # If all of the above succeeds, enable plotting of predicted sowing/harvest
        add_sow_harv = True
    except Exception as e:
        print(f"Error processing combination S1: {s1_obs}, S2: {s2_obs}, Error: {e}")
        continue
    # ---------- colors ----------
    color_s1 = cmap_s1(norm_s1(s1_obs))
    color_s2 = cmap_s2(norm_s2(s2_obs))
    color_h  = cmap_hybris(norm_h(s1_obs + s2_obs))
    if s1_obs == max_s1 and s2_obs == max_s2: # if we have all observations, use black for better visibility
        color_h = 'black'
        color_s2 = 'black'
        color_s1 = 'black'

    hybris_sub = merge_with_GT(hybris_sub, field) #combine everything in one long dataframe

    plot_hybris(hybris_sub[(hybris_sub["Date"] > start_date) & (hybris_sub["Date"] < end_date)],
                add_pred_sow_harv = False,
                plot_tillages=False,
                plot_dormant=False,
                add_groundtruth=True,
                ax = axes[0], date_col = "Date",
                color = color_h, # color based on number of observations
                legend = False,
                alpha_raw=0, alpha_smooth=1
                )
    plot_hybris(hybris_sub[(hybris_sub["Date"] > start_date) & (hybris_sub["Date"] < end_date)],
            add_pred_sow_harv = add_sow_harv,
            plot_tillages=False,
            plot_dormant=False,
            add_groundtruth=False,
            ax = axes[1], date_col = "Date",
            color = color_h, # color based on number of observations
            legend = False,
            alpha_raw=1, alpha_smooth=0
            )
    #add BSI and VV/VH time series to the plot
    plot_time_series(1-s2_sub[s2_band], s2_sub['date'], color=color_s2, title="- " +s2_band + f" (n={len(s2_sub)})", add=True,
                    show=False, alpha=0.8, marker='o', ax=axes[2], legend = False)
    plot_time_series(1-s1_sub[s1_band], s1_sub['date'], color=color_s1, title="- VV/VH" + f" (n={len(s1_sub)})",
                    add=True, show=False, alpha=0.8, marker='o', ax=axes[3], legend=False)

fig.colorbar(sm_h, ax=axes[0], label="Total observations (S1 + S2)                      ")
fig.colorbar(sm_h, ax=axes[1], label="")
fig.colorbar(sm_s2, ax=axes[2], label="Sentinel-2 observations")
fig.colorbar(sm_s1, ax=axes[3], label="Sentinel-1 observations")
axes[0].set_title("HyBRIS smoothed response to data availability")
axes[1].set_title("HyBRIS unsmoothed response to data availability")
axes[2].set_title("Optical signal (BSI)")
axes[3].set_title("Radar signal (VV/VH)")
plt.tight_layout()
plt.show()