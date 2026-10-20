import json
import os

## FUNCTIONS TO DOWNLOAD TIME SERIES DATA FROM GOOGLE EARTH ENGINE
def _require_ee():
    """Import and return the optional Google Earth Engine client.

    Returns
    -------
    module
        The imported :mod:`ee` module.

    Raises
    ------
    ImportError
        If the optional ``gee`` dependency is not installed.
    """
    try:
        import ee #to use the Google Earth Engine library, you need to authenticate and initialize it first (ee.Initialize()/ee.Authenticate())
    except ImportError as e:
        raise ImportError(
            "This function needs the Earth Engine API: pip install 'hybris[gee]'"
        ) from e
    return ee

# Function to transform a shapefile into a ee Feature Collection.
def shp_to_ee(shapefile_path):
    """
    Takes a shapefile path as input and loads it as a ee Feature collection (a google earth engine object)
    """
    ee = _require_ee()

    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "This function needs GeoPandas: pip install 'hybris[examples]'"
        ) from e

    # Step 1: Load the shapefile using GeoPandas
    geopandas_df = gpd.read_file(shapefile_path)

    # Step 2: Reproject to WGS84 (if not already in EPSG:4326)
    geopandas_df = geopandas_df.to_crs(epsg=4326)

    # Step 3: Convert to GeoJSON
    geojson = json.loads(geopandas_df.to_json())

    # Step 4: Convert GeoJSON to Google Earth Engine FeatureCollection
    gee_fc = ee.FeatureCollection(geojson)

    return gee_fc

def get_S2_one_field(fields, field_id, bands, start_date, end_date, cloud_filter, output_dir, id_column = 'ID', download = True):
    """Download Sentinel-2 observations and field-level band medians.

    The function filters Sentinel-2 SR imagery by geometry, date, cloud
    percentage, and Cloud Score Plus quality, then reduces each image over
    the field geometry. Earth Engine must be authenticated and initialized.

    Parameters
    ----------
    fields : ee.FeatureCollection
        Collection containing the field geometries; The field_id is filtered.
    field_id : object
        Identifier written to each output record and filename.
    bands : sequence of str
        Sentinel-2 bands to select and reduce.
    start_date, end_date : str
        Inclusive start and exclusive end dates accepted by Earth Engine.
    cloud_filter : float
        Maximum ``CLOUDY_PIXEL_PERCENTAGE``.
    output_dir : path-like
        Directory for the CSV when ``download`` is true.
    download : bool, default True
        Return a CSV path when true; otherwise return the Earth Engine result.

    Returns
    -------
    str or list
        Output CSV path, or the list of per-image records when not downloading.
    """
    ee = _require_ee()
    field = (ee.FeatureCollection(fields)
            .filter(ee.Filter.eq(id_column, field_id))
            .first())

    # Initialize cloud score collection
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    csPlusBands = csPlus.first().bandNames()

    def maskLowQA(image):
        """Mask pixels whose Cloud Score Plus score is below 0.5."""
        qaBand = 'cs'
        clearThreshold = 0.5
        mask = image.select(qaBand).gte(clearThreshold)
        return image.updateMask(mask)

    # Initialize Sentinel-2 collection
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(field.geometry())
                .filterDate(start_date, end_date)
                .select(bands)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
                .linkCollection(csPlus, csPlusBands)
                .map(maskLowQA))
    
    def medianBands(image):
        """Compute field medians and attach a serializable output record."""
        medianValues = (image
                        .select(bands)
                        .reduceRegion(
                            reducer=ee.Reducer.median(), 
                            geometry=field.geometry(),
                            scale=10,
                            maxPixels=1e8
                        )
                        .combine({band: -9999 for band in bands}, overwrite=False))
        formattedValues = [medianValues.getNumber(band).format('%.4f') for band in bands]
        date = image.date().format('YYYY-MM-dd')
        output = [date, field_id] + formattedValues
        return image.set('output', output)

    timeSeries = (collection.map(medianBands))
    result = timeSeries.aggregate_array('output').getInfo()

    # Build header row: Date, FieldID, then band medians
    header_df = ["date", id_column]
    for band in bands:
        header_df.extend([f"{band}"])

    if download:
        filename = os.path.join(output_dir, f'Sentinel2_{field_id}.csv')
        with open(filename, 'w') as out_file:
            print(','.join(header_df), file=out_file)
            for items in result:
                line = ','.join([str(item) for item in items])
                print(line, file=out_file)
        print("Downloaded Sentinel 2")
        return filename
    else:
        return result
    
def get_S1_one_field(fields, field_id, bandsusedS1, start_date, end_date, output_dir, id_column = "ID", download = True):
    """Download preprocessed Sentinel-1 field-level band medians.

    Sentinel-1 IW, 10 m imagery is preprocessed with ``gee_s1_ard`` using
    border-noise correction, speckle filtering, and terrain flattening.
    Earth Engine must be authenticated and initialized before calling this
    function.

    Parameters
    ----------
    field : ee.FeatureCollection
        Collection containing the field geometries; The specified field (using field_id) is select.
    field_id : object
        Identifier written to each output record and filename.
    bandsusedS1 : sequence of str
        Sentinel-1 bands to reduce after preprocessing.
    start_date, end_date : str
        Date range accepted by Earth Engine.
    output_dir : path-like
        Directory for the CSV when ``download`` is true.
    download : bool, default True
        Return a CSV path when true; otherwise return the result records.

    Returns
    -------
    str or list
        Output CSV path, or the list of per-image records when not downloading.
    """
    ee = _require_ee()

    # Functions to preprocess Sentinel 1 images:
    #Refer to Mulissa et al. 2021 https://doi.org/10.3390/rs13101954, the repository was cloned from https://github.com/adugnag/gee_s1_ard/tree/main/python-api
    from hybris.preprocessS1 import wrapper  # gee_s1_ard preprocessing 

    #Filter fields to the field of interest
    field = (ee.FeatureCollection(fields)
        .filter(ee.Filter.eq(id_column, field_id))
        .first())

    #Load collection of Sentinel-1 images, filtered by date, bands, and field geometry
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                .filterDate(start_date, end_date)
                .select(bandsusedS1)
                .filterBounds(field.geometry())
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.eq('resolution_meters', 10))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    
    # Define parameters for Sentinel-1 preprocessing
    parameter = {
        'COLLECTION': collection,
        'START_DATE': start_date,
        'STOP_DATE': end_date,
        'POLARIZATION': 'VVVH',
        'ORBIT': 'BOTH',
        'ROI': field.geometry(),
        'APPLY_BORDER_NOISE_CORRECTION': True,
        'APPLY_SPECKLE_FILTERING': True,
        'SPECKLE_FILTER_FRAMEWORK': 'MONO',
        'SPECKLE_FILTER': 'LEE',
        'SPECKLE_FILTER_KERNEL_SIZE': 5,
        'SPECKLE_FILTER_NR_OF_IMAGES': 5,
        'APPLY_TERRAIN_FLATTENING': True,
        'DEM': ee.Image('USGS/SRTMGL1_003'),
        'TERRAIN_FLATTENING_MODEL': 'VOLUME',
        'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
        'FORMAT': 'DB',
        'CLIP_TO_ROI': False,
        'SAVE_ASSET': False,
        'ASSET_ID': "users/your_username/your_asset_name"  # Replace with your desired asset path
    }

    # Preprocessed Sentinel-1 collection
    collection = wrapper.s1_preproc(parameter)
    
    #calculate median
    def medianBands_S1(image):
        """Compute field medians and attach orbit metadata to the record."""
        # Reduce the region for all bands at once
        medianValues = (image
                        .select(bandsusedS1)  # Select only the bands of interest
                        .reduceRegion(
                            reducer=ee.Reducer.median(),
                            geometry=field.geometry(),
                            scale=10,  # Adjust the scale as needed
                            maxPixels=1e8  # Increase if working with large geometries
                        )
                        .combine({band: -9999 for band in bandsusedS1}, overwrite=False))  # Set default values

        # Format median values for output
        formattedValues = [medianValues.getNumber(band).format('%.4f') for band in bandsusedS1]

        # Add the date
        date = image.date().format('YYYY-MM-dd')

        # Add the Sentinel 1 specifics
        orbit = image.get('orbitProperties_pass')
        orbitNumber = ee.Number(image.get('relativeOrbitNumber_start')) 
                                                            
        # Combine all output: [date, fieldID, band1_median, band2_median, ...]
        output = [date, field_id, orbit, orbitNumber] + formattedValues

        return image.set('output', output)
    
    timeSeries = (collection.map(medianBands_S1))
    
    result = timeSeries.aggregate_array('output').getInfo()

    # Build header row: Date, FieldID, "orbit", "orbitNumber", then band medians
    header_df = ["date", id_column, "orbit", "orbitNumber"]
    for band in bandsusedS1:
        header_df.extend([f"{band}"])

    if download:
        filename = os.path.join(output_dir, f'Sentinel1_{field_id}.csv')
        with open(filename, 'w') as out_file:
            print(','.join(header_df), file=out_file)
            for items in result:
                line = ','.join([str(item) for item in items])
                print(line, file=out_file)
        print("Done with Sentinel 1")
        return filename
    else:
        return result 