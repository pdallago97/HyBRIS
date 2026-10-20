# HyBRIS

[![PyPI version](https://img.shields.io/pypi/v/hybris.svg)](https://pypi.org/project/hybris/) [![License](https://img.shields.io/pypi/l/hybris.svg)](https://github.com/pdallago97/HyBRIS/blob/main/LICENSE)

This package offers functions and tools to download and handle Sentinel-1 and Sentinel-2 time series, and calculate hybrid indices.

## Optical, radar, and hybrid indices to detect farming practices in Europe

Author: Paolo Dal Lago

Laboratory of Geo-Information Science and Remote Sensing (GRS), Wageningen University & Research

Please refer to DOI for more information: https://www.sciencedirect.com/science/article/pii/S0034425726003238

## Install

Using pip:
```bash
pip install hybris
```

Or using pixi:
```bash
pixi add --pypi hybris
```

## Requirements
To correctly set up your Google Earth Engine project, it needs to be authenticated and initialized.

- Google Earth Engine authentication via
https://developers.google.com/earth-engine/apidocs/ee-authenticate

- Google Earth Engine initialization via
https://developers.google.com/earth-engine/apidocs/ee-initialize

To download Sentinel-1 and -2 time series, Earth Engine must be installed. To install the Google Earth Engine api:

```bash
pip install "hybris[gee]"
```

To use `shp_to_ee` with local shapefiles, install GeoPandas as well:

```bash
pip install "hybris[gee,examples]"
```

- Sentinel-1 preprocessing package:
The Sentinel-1 preprocessing pipeline uses the `gee_s1_ard` Python API under the hood; see its instructions in the function documentation (https://github.com/adugnag/gee_s1_ard).

## Quickstart

The example below reads prepared field-level CSV files, creates optical and
radar indices, fuses them into a daily series, and detects farming events.

```python
import hybris

bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
bands_s1 = ["VV", "VH", "angle"]
s2 = hybris.add_vis(hybris.openSentinel2file("examples/data/Sentinel2_example.csv", bands_s2))
s1 = hybris.add_vis_radar(hybris.openSentinel1file("examples/data/Sentinel1_example.csv", bands_s1))
hybrid = hybris.calculate_hybris_vectorized(s1, s2)
maxima = hybris.find_maxima(hybrid)
minima = hybris.find_minima(hybrid)
```

For a complete workflow including orbit selection, ground truth, and plots, run [`examples/hybris_example.py`](https://github.com/pdallago97/HyBRIS/blob/main/examples/hybris_example.py).

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
tests/          Tests
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

The Sentinel-1 ARD preprocessing is adapted from gee_s1_ard:

Source: https://github.com/adugnag/gee_s1_ard (python-api folder)
Author: Adugna Mullissa et al., MIT License — see [their license](src/hybris/preprocessS1/THIRD_PARTY_LICENCE).
Reference: Mullissa et al. (2021), https://doi.org/10.3390/rs13101954
