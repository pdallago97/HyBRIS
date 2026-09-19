from importlib.metadata import version, PackageNotFoundError

from .utils import (
    openSentinel2file, openSentinel1file, selectOrbit,
    add_vis, add_vis_radar, normalize_percentiles,
    daily_index_with_contributions_vectorized, calculate_hybris_vectorized,
    find_maxima, find_minima, growing_seasons, add_tillages,
    add_predictions, merge_with_GT, validate_predictions,
    filter_by_date, plot_hybris, plot_time_series,
    get_S2_one_field, get_S1_one_field,
)

try:
    __version__ = version("hybris")   # must match the name in pyproject.toml
except PackageNotFoundError:
    __version__ = "unknown"