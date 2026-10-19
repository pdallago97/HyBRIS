import os
import hybris

##### We use Google Earth Engine to download satellite data
# A Google Earth Engine project is needed to continue. You can follow https://courses.spatialthoughts.com/gee-sign-up.html for setting it up.
# Insert your ee-project
print("Is gee initialied? ", hybris.is_initialized())   # should print False
ee = hybris.initialize_ee('ee-yourprojectname')
# Follow the Authentication instructions and copy paste the key when required.
print("Is gee initialied? ", hybris.is_initialized())   # should now be True

####################################### BANDS ################################################
# Select the Sentinel-1 bands
bandsusedS1 = ['VV', 'VH', 'angle']
# Select the Sentinel-2 bands
bandsusedS2 = ['B1', 'B2', 'B3', 'B4','B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

####################################### CLOUD FILTER ################################################
# Define the cloud filter percentage.
# Sentinel-2 images with a cloud percentage higher than the CLOUD_FILTER value will be discarded
CLOUD_FILTER = 50

####################################### TIME ################################################
# Note that Sentinel-1 and Sentinel-2 data are available only after their launch in 2014 and 2015
# Sentinel-1 after 2014 (first image in GEE 2014-10-03),
# Sentinel-2 after 2015 (first image in GEE 2015-07-04) 

# Define the starting date for satellite image download (the desired start of the time series)
start = ee.Date("2026-01-01")
# Define the end date for satellite image download (the desired end of the time series)
end = ee.Date("2026-01-31")

####################################### SPACE ################################################
# Load your polygons!
# This can either be a shapefile (.shp) or a geopackage (.gpkg)
shapefile_path = 'path/to/your/shapefile.shp'
fields = hybris.shp_to_ee(shapefile_path)

# Alternatively, you can also write the path of your GEE asset, if it is already loaded in your GEE account.
# fields = ee.FeatureCollection('path/to/gee/asset')

####################################### OUTPUT ################################################
# Define the directory where results (time-series) should be stored. Make sure the directory exists
output_dir = r"examples/my_timeseries"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

####################################### DOWNLOAD ################################################
hybris.get_S1_one_field(fields, 1, bandsusedS1, start, end, output_dir)
hybris.get_S2_one_field(fields, 1, bandsusedS2, start, end, CLOUD_FILTER, output_dir)
