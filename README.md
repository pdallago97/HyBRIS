**Optical, radar, and hybrid indices to detect farming practices in Europe**

Paolo Dal Lago, Lammert Kooistra, Nandika Tsendbazar, Kirsten de Beurs

*Laboratory of Geo-Information Science and Remote Sensing (GRS), Wageningen University & Research*

Please refer to DOI for more information: https://www.sciencedirect.com/science/article/pii/S0034425726003238

-------------------------------------------------------------------------------------------------------------
*Summary*

A temporally weighted mean is used to integrate SAR and optical indices into a daily hybrid index. Weighted averaging emphasizes coincident temporal patterns, fills data gaps emerging from irregular acquisitions, and helps reduce sensor-specific noise. This integration aims at stabilizing complementary signals, not to estimate a physically true variable. A new index is obtained, but not a new physical unit, as both optical and radar indices are inherently unitless. Local minima and maxima of the time series are used to detect farming practices across European sites without relying on complex line-fitting functions. This field-level methodology accounts for multiple growing seasons, achieving strong generalizability across indices, years, locations, and crop types. Building on previous pixel-level approaches for monitoring bare soil periods (Lobert et al., 2025; Mzid et al., 2021), the developed methodology allows the calculation of Dormant Days, as a novel field-level, multi-sensor metric for quantifying dormant periods over long time series. 

-------------------------------------------------------------------------------------------------------------
*Folder structure:*


HyBRIS_utils.py : contains all the functions to download Sentinel 1 and Sentinel 2 data, calculate hybrid indices, and apply the minMax method to detect farming practices.

HyBRIS_example.py : runs the code to calculate HyBRIS for an example field, detects sowing, harvests, and tillage dates, and plot results.

hybrid_index.py : runs the code to calculate a generic hybrid index for an example field and plot results.

HyBRIS_playground.ipynb: Jupyter notebook to load and calculate HyBRIS. Here, the Sentinel-1 and Sentinel-2 contributions per day are explored and plotted.

Dataset/: contains (part of) the dataset used for the research paper, saved as .csv. The tillage (1), sowing (2), and harvest (3) dates were recorded for three farms, one in Italy (Vallevecchia), and two in the Netherlands (Valthermond, and Unifarm in Wageningen). The crop type is reported as well. The georeferenced boundary of each field are available as a geopackage (.gpkg). This can be linked to the .csv through the unique ID column.

Example/: contains the Sentinel 1 and 2 time series and the ground truth for an example field. These files are used to run the example code

-------------------------------------------------------------------------------------------------------------
*Calculation of a daily hybrid index*

Here, two indices (one optical, one radar) sensitive to soil roughness and bare soil exposure are combined to detect sowing, harvesting, and tillage events: the inverted Bare Soil Index (1 - BSI) from Sentinel-2 (a) and the VH/VV index from Sentinel-1 (b). The normalization step ensures the values of BSIinv and VH/VV to be within the [0, 1] range.

With this combination, we developed the daily Hybrid Bare Soil Radar Index (HyBRIS) (c). To calculate HyBRIS for each day, a temporal window of ±12 days was adopted.  This window size ensures the inclusion of at least two radar acquisitions from the same orbit before and after the target day, given the 12-day revisit cycle of each orbit of Sentinel 1, and increases the likelihood of incorporating cloud-free optical observations during persistently cloudy periods.

The Sentinel-1 and -2 observations acquired within this window are considered, and aggregated with a weighted mean. Expanding the window had minimal impact on the aggregated index values, as the weighting mean reduces the influence of observations further from the target day.

This way, images acquired closer to the target day have higher weights and contribute more to the fused index, while images acquired further from each day have lower weights. 

Next, a smoothed time series was calculated for HyBRIS and for each optical and radar index using a centered rolling mean with a window of ±15 days. Both smoothed and unsmoothed time series were then used to detect farming practices.

!<img width="11693" height="5529" alt="Figure2" src="https://github.com/user-attachments/assets/6e7cb8b0-f6c0-493f-a0cd-da521e2e7a38" />

Note that the considered length of the time series influences the amplitude of the time series itself. This is due to the normalization process done with percentiles (0.02-0.98) within the observation period.

-------------------------------------------------------------------------------------------------------------
*How to cite*

Paolo Dal Lago, Lammert Kooistra, Nandika Tsendbazar, Kirsten de Beurs,
Optical, radar, and hybrid indices to detect farming practices in Europe,
Remote Sensing of Environment, Volume 344, 2026, 115553, ISSN 0034-4257,
https://doi.org/10.1016/j.rse.2026.115553.
