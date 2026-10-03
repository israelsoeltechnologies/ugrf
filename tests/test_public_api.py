import ugrf


def test_public_api_is_small():
    assert ugrf.__version__ == "0.1.0"
    assert ugrf.__all__ == ["TSB", "RenewalForecaster", "UtilityGatedRenewal"]
