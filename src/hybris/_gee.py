"""Lazy import and initialization helpers for Google Earth Engine (ee).

Earth Engine is an optional dependency of hybris (install with
`pip install "hybris[gee]"`), and every call also needs a Google Cloud
project the caller owns or has permission to use -- `ee.Initialize()`
no longer works with no arguments for most accounts. See
https://developers.google.com/earth-engine/guides/python_install.
Non-commercial projects should also have a quota tier selected in the
Earth Engine console; unselected projects default to the lowest tier.
"""

_initialized = False


def _require_ee():
    """Import and return the ``ee`` module.

    Returns
    -------
    module
        The imported ``ee`` module.

    Raises
    ------
    ImportError
        If ``earthengine-api`` is not installed.
    """
    try:
        import ee
    except ImportError as e:
        raise ImportError(
            "This function needs the Earth Engine API: "
            "pip install 'hybris[gee]'"
        ) from e
    return ee


def is_initialized():
    """Whether `initialize_ee` has successfully run in this session.

    Returns
    -------
    bool
    """
    return _initialized


def initialize_ee(project, force=False):
    """Authenticate (if needed) and initialize Earth Engine for this session.

    Call this once, near the top of your script or notebook, before using
    any hybris function that downloads data from Earth Engine. If no
    credentials are stored yet, this opens the usual browser-based
    `ee.Authenticate()` flow.

    Parameters
    ----------
    project : str
        A Google Cloud project you own or have permission to use with
        Earth Engine.
    force : bool, default False
        Re-run initialization even if this session already succeeded.

    Returns
    -------
    module
        The initialized ``ee`` module (so you can also do
        ``ee = hybris.initialize_ee('my-project')`` if you want it directly).

    Examples
    --------
    >>> import hybris
    >>> hybris.initialize_ee('my-project')
    >>> s2 = hybris.get_S2_one_field(...)
    """
    global _initialized
    ee = _require_ee()

    if _initialized and not force:
        return ee

    try:
        ee.Initialize(project=project)
    except Exception:
        # Most likely: no stored credentials yet, or they expired.
        ee.Authenticate()
        ee.Initialize(project=project)

    _initialized = True
    return ee