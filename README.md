# HyBRIS

[![PyPI version](https://img.shields.io/pypi/v/hybris.svg)](https://pypi.org/project/hybris/)
[![License](https://img.shields.io/pypi/l/hybris.svg)](LICENSE)

## Optical, radar, and hybrid indices to detect farming practices in Europe

This package offers functions and tools to handle Sentinel-1 and Sentinel-2 time series, and calculate hybrid indices.

Paolo Dal Lago, Lammert Kooistra, Nandika Tsendbazar, Kirsten de Beurs

Laboratory of Geo-Information Science and Remote Sensing (GRS), Wageningen University & Research

Please refer to DOI for more information: https://www.sciencedirect.com/science/article/pii/S0034425726003238

## Install

```bash
pip install hybris
```

To download Sentinel-1 and -2 time series, Earth Engine must be installed. To install the Google Earth Engine api:

```bash
pip install "hybris[gee]"
```
When your Google Earth Engine project is setup, it needs to be authenticated ([![ee.Authenticate()](https://developers.google.com/earth-engine/apidocs/ee-authenticate)]) and initialized ([![ee.Initialize()](https://developers.google.com/earth-engine/apidocs/ee-initialize)]).

 The Sentinel-1 preprocessing function uses the `gee_s1_ard` Python API; see its
installation instructions in the function documentation [![gee_s1_ard](https://github.com/adugnag/gee_s1_ard)].

## Quickstart

The example below reads prepared field-level CSV files, creates optical and
radar indices, fuses them into a daily series, and detects farming events.

```python
from hybris import (
    openSentinel1file, openSentinel2file, add_vis, add_vis_radar,
    calculate_hybris_vectorized, find_maxima, find_minima,
)

bands_s2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
bands_s1 = ["VV", "VH", "angle"]
s2 = add_vis(openSentinel2file("examples/data/Sentinel2_example.csv", bands_s2))
s1 = add_vis_radar(openSentinel1file("examples/data/Sentinel1_example.csv", bands_s1))
hybris = calculate_hybris_vectorized(s1, s2)
maxima = find_maxima(hybris)
minima = find_minima(hybris)
```

For a complete workflow including orbit selection, ground truth, validation,
and plots, run [`examples/hybris_example.py`](examples/hybris_example.py).

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
tests/          Package smoke tests
```

## Documentation

- [API and supporting documentation](docs/)
- [Runnable examples](examples/)
- [Research paper](https://doi.org/10.1016/j.rse.2026.115553)

## Citation

Paolo Dal Lago, Lammert Kooistra, Nandika Tsendbazar, Kirsten de Beurs,
"Optical, radar, and hybrid indices to detect farming practices in Europe,"
*Remote Sensing of Environment*, volume 344, 2026, 115553, DOI: https://doi.org/10.1016/j.rse.2026.115553.

## License

HyBRIS is released under the [MIT License](LICENSE).
