import os
import hybris

##### We use Google Earth Engine to download satellite data
print("Is gee initialied? ", hybris.is_initialized())   # should print False

# IMPORTANT: A Google Earth Engine project is needed to continue. You can follow https://courses.spatialthoughts.com/gee-sign-up.html for setting it up.
# Make sure to insert your ee-project
ee = hybris.initialize_ee('ee-yourprojectname') #Insert the name of your existing ee-project
# Follow the Authentication instructions and copy paste the key when required.

print("Is gee initialied? ", hybris.is_initialized())   # should now be True

####################################### BANDS ################################################
# Select the Sentinel-1 bands
bandsusedS1 = ['VV', 'VH', 'angle']
# Select the Sentinel-2 bands
bandsusedS2 = ['B1', 'B2', 'B3', 'B4','B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

####################################### CLOUD FILTER ################################################
# Define the cloud filter percentage for Sentinel-2.
# Sentinel-2 images with a cloud percentage higher than the CLOUD_FILTER value will be discarded
CLOUD_FILTER = 75

# Clouds are removed from the remaining images following CloudScore+ (Pasquarella et al. 2023)

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
#fields = ee.FeatureCollection('path/to/gee/asset')

#specify the name of the id_column in your collection
id_column = "ID"

# Specify the field_id to download (often an integer). The first field is taken in this example
field_id = fields.aggregate_array(id_column).getInfo()[0]

print("Downloading field:", field_id)

####################################### OUTPUT ################################################
# Define the directory where results (time-series) should be stored.
output_dir = r"examples/my_timeseries"

# Makes sure the directory exists, otherwise it creates a new folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

####################################### DOWNLOAD ################################################
#Sentinel-1 and Sentinel-2 data are preprocessed and downloaded as .csv time series (one row per date, one column per band)
# preprocessing is explained in detail in https://doi.org/10.1016/j.rse.2026.115553
hybris.get_S1_one_field(fields, field_id, bandsusedS1, start, end, output_dir, id_column, True)
hybris.get_S2_one_field(fields, field_id, bandsusedS2, start, end, CLOUD_FILTER, output_dir, id_column, True)
