**HyBRIS: a Hybrid Bare Soil Radar Index to detect farming practices in Europe**

Paolo Dal Lago, Lammert Kooistra, Nandika Tsendbazar, Kirsten de Beurs

*Laboratory of Geo-Information Science and Remote Sensing (GRS), Wageningen University & Research*

Please refer to DOI for more information: https://doi.org/10.2139/ssrn.5580396 #paper under review#

-------------------------------------------------------------------------------------------------------------
Folder structure:


HyBRIS_utils.py : contains all the functions to calculate a Sentinel 1 and Sentinel 2 hybrid index

HyBRIS_example.py : runs the code to calculate HyBRIS for an example field and plot results

Dataset/: contains (part of) the dataset used for the research paper, saved as .csv

Example/: contains the Sentinel 1 and 2 time series and the ground truth for an example field. These files are used to run the example code

-------------------------------------------------------------------------------------------------------------

Summary of methodology

We combined indices sensitive to soil roughness and bare soil exposure (VV/VH, BSI) to detect sowing, harvesting, and tillage. We combined the Bare Soil Index (BSI) derived from Sentinel 2 and the VV/VH ratio derived from Sentinel 1 into the daily Hybrid Bare Soil and Radar Index (HyBRIS) (Figure below, a). The normalization step ensures the values of BSI and VV/VH to be within the [0, 1] range. To calculate HyBRIS for each day, a temporal window of ±30 days was adopted.  X ̅_d  is the weighted mean of the radar and optical observations that fall within this defined timeframe. This window size ensures the inclusion of at least two radar acquisitions from the same orbit before and after the target day, given the 12-day revisit cycle of each orbit of Sentinel 1, and increases the likelihood of incorporating cloud-free optical observations during persistently cloudy periods. The Sentinel 1 and 2 observations acquired within this window are considered (n), and aggregated with a weighted mean. Expanding the window beyond ±30 days had minimal impact on the aggregated index values, as the weighting mean reduces the influence of observations further from the target day. The weights (wi) of each Sentinel observation are calculated based on the inverse of the absolute difference in days between the target day (d) and the date of acquisition of the Sentinel image (i).

![image](https://github.com/user-attachments/assets/960f70d4-cd5b-41f6-8327-59fe923ea94a)

This way, images acquired closer to the target day have higher weights and contribute more to the fused index, while images acquired further from each day have lower weights. 
Thus, the weighted mean becomes:

![image](https://github.com/user-attachments/assets/94ad335f-03d1-4f03-9099-00dd3daea271)

where:
	X ̅_d is the value of the fused daily index at day d
	n is the total number of images present in the temporal window d±30
	X_i is the value of the Sentinel index at image date i, either BSI or VV/VH
	w_i is the weight assigned to each Sentinel observation
	The denominator ∑_i^n▒w_i  is the sum of all weights within the temporal window
BSI and VV/VH are sensitive to bare soil and show high values in the absence of vegetation. The weighted mean is subtracted from 1 to resemble the shape of classic vegetation indices (Figure below, b), with high values representing peaks of vegetation and low values bare soils:
HYBRIS=1- X ̅_d


![image](https://github.com/user-attachments/assets/4636ba3f-c3d8-436c-901b-5cdde1ebccb1)

Note that the considered length of the time series influences the amplitude of the time series itself. This is due to the normalization process done with percentiles (0.02-0.98) within the observation period.
