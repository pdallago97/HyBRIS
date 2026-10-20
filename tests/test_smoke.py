def test_import():
    import hybris
    assert hasattr(hybris, "calculate_hybris_vectorized")


def test_plot_time_series_creates_axes_without_provided_axes():
    import matplotlib.pyplot as plt
    import pandas as pd
    import hybris

    ax = hybris.plot_time_series(
        [1.0, 2.0], pd.date_range("2024-01-01", periods=2), show=False
    )
    assert ax is not None
    plt.close(ax.figure)