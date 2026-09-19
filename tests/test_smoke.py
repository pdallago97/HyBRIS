def test_import():
    import hybris
    assert hasattr(hybris, "calculate_hybris_vectorized")