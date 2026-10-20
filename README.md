# HyBRIS

[![PyPI version](https://img.shields.io/pypi/v/hybris.svg)](https://pypi.org/project/hybris/) [![License](https://img.shields.io/pypi/l/hybris.svg)](https://github.com/pdallago97/HyBRIS/blob/main/LICENSE)

This package offers functions and tools to download and handle Sentinel-1 and Sentinel-2 time series, and calculate hybrid indices.

## Optical, radar, and hybrid indices to detect farming practices in Europe

Author: Paolo Dal Lago

Laboratory of Geo-Information Science and Remote Sensing (GRS), Wageningen University & Research

Please refer to the [research paper](https://doi.org/10.1016/j.rse.2026.115553) for more information.

## Install

Using pip:
```bash
pip install hybris
```

Or using pixi:
```bash
pixi add --pypi hybris
```

## Install optional dependencies
To download Sentinel-1 and -2 time series, Earth Engine must be installed. By installing the Google Earth Engine api and GeoPandas, your own shapefiles can be used for time-series download (through `shp_to_ee()`). To install these optional dependencies, run:

Using pip:
```bash
pip install hybris[gee]
```

Or using pixi:
```bash
pixi add --pypi hybris[gee]
```

## Requirements

## Google Earth Engine
A Google Earth Engine project is needed to download Sentinel-1 and -2 time series for your own georeferenced fields. You can follow [`this tutorial`](https://courses.spatialthoughts.com/gee-sign-up.html) to set up your Google Earth Engine project.

Then, follow the [`examples/download_ts_example.py`](https://github.com/pdallago97/HyBRIS/blob/main/examples/download_ts_example.py) to make sure your project is authenticated and initialized:

```python
import hybris
# Make sure to insert your ee-project
ee = hybris.initialize_ee('ee-yourprojectname') #Insert the name of your existing ee-project
print(hybris.is_initialized())   # should now be True
```

More info:
- [`Google Earth Engine authentication`](https://developers.google.com/earth-engine/apidocs/ee-authenticate)
- [`Google Earth Engine initialization`](https://developers.google.com/earth-engine/apidocs/ee-initialize)

## Load your own polygons!
This can be a shapefile (.shp) or a geopackage (.gpkg)

```python
shapefile_path = 'local/path/to/your/shapefile.shp'
fields = hybris.shp_to_ee(shapefile_path) #fields is converted to an Earth Engine object: ee.FeatureCollection
```

Alternatively, you can also write the path of your GEE asset, if it is already loaded in your GEE account.

```python
fields = ee.FeatureCollection('path/to/your/gee/asset')
```

## Quickstart

The example below reads prepared field-level CSV files, creates optical and
radar indices, fuses them into a daily series, and detects farming events.

```python
import hybris

#Select bands of interest for Sentinel-2 and Sentinel-1
bandsusedS2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
bandsusedS1 = ["VV", "VH", "angle"]

#Open Sentinel-2 file, remove NA, and add Vegetation Indices
s2 = hybris.add_vis(hybris.openSentinel2file("examples/data/Sentinel2_example.csv", bandsusedS2))
#Open Sentinel-1 file, remove NA, and add Vegetation Indices
s1 = hybris.add_vis_radar(hybris.openSentinel1file("examples/data/Sentinel1_example.csv", bandsusedS1))

#Calculate HyBRIS
hybrid = hybris.calculate_hybris_vectorized(s1, s2)

#Apply minMax method
maxima = hybris.find_maxima(hybrid)
minima = hybris.find_minima(hybrid)

# Identify growing seasons based on predicted minima
g_seasons = hybris.growing_seasons(maxima, minima)
# Add tillages to predictions
predictions = hybris.add_tillages(g_seasons, minima)
#Merge with predictions
hybrid = hybris.add_predictions(hybrid, predictions)

```

For a complete workflow including orbit selection, ground truth, and plots, run [`examples/hybris_example.py`](https://github.com/pdallago97/HyBRIS/blob/main/examples/hybris_example.py).

## Download the time series
Sentinel-1 and Sentinel-2 data are preprocessed and downloaded as .csv time series: one row per date, one column per band. The preprocessing and methodology is explained in detail in the [research paper](https://doi.org/10.1016/j.rse.2026.115553)

The download of preprocessed time series for a given field is wrapped in one line of code:
```python
hybris.get_S1_one_field(fields, field_id, bandsusedS1, start, end, output_dir, id_column, True)
hybris.get_S2_one_field(fields, field_id, bandsusedS2, start, end, CLOUD_FILTER, output_dir, id_column, True)
```

For a complete workflow, follow [`examples/download_ts_example.py`](https://github.com/pdallago97/HyBRIS/blob/main/examples/download_ts_example.py).

## Method summary

The HyBRIS methodology combines one optical and one radar index into a daily, field-level time series.
Here, Sentinel-2 observations of the Bare Soil Index (BSI) and Sentinel-1
VH/VV observations are normalized and aggregated with a temporally weighted
mean within a +/-12-day window. Closer observations have greater influence,
which helps bridge irregular acquisitions and sensor-specific gaps. A
centered rolling mean is used alongside the unsmoothed series to detect local
minima and maxima associated with sowing, harvest, and tillage.

The index is unitless and is intended to stabilize complementary signals, not
to estimate a physically true variable. Because percentile normalization is
computed over the selected observation period, the time range affects the
index amplitude.

![Calculation of HyBRIS from inverted BSI (Sentinel-2) and VH/VV (Sentinel-1)](https://raw.githubusercontent.com/pdallago97/HyBRIS/main/docs/images/figure2.png)

## Repository layout

```text
src/hybris/     Package implementation and public API
examples/       Runnable scripts and example input data
Dataset/        Research dataset and field-level time series
docs/           Figures and supporting documentation
tests/          Test for correctly importing the package
```

## Examples

- [Runnable examples](https://github.com/pdallago97/HyBRIS/blob/main/examples/)

## Research paper dataset

- [Dataset](https://github.com/pdallago97/HyBRIS/blob/main/Dataset/)

## More information

- [Research paper](https://doi.org/10.1016/j.rse.2026.115553)

## Citation

Paolo Dal Lago, Lammert Kooistra, Nandika Tsendbazar, Kirsten de Beurs,
"Optical, radar, and hybrid indices to detect farming practices in Europe,"
*Remote Sensing of Environment*, volume 344, 2026, 115553, DOI: https://doi.org/10.1016/j.rse.2026.115553.

## License

HyBRIS is released under the [GNU Affero General Public License v3 or later (AGPL-3.0-or-later)](LICENSE).

## Acknowledgements

The Sentinel-1 preprocessing pipeline is adapted from the `gee_s1_ard` Python API, which is used under the hood; see its [`instructions`](https://github.com/adugnag/gee_s1_ard).

Source: https://github.com/adugnag/gee_s1_ard (python-api folder)

Author: Adugna Mullissa et al., MIT License — see [their license](src/hybris/preprocessS1/THIRD_PARTY_LICENCE).

Reference: Mullissa et al. (2021), https://doi.org/10.3390/rs13101954
