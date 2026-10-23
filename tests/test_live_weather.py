"""Regression tests for NOAA METAR parsing (real API returns strings like '6+')."""

from src.data import live_weather
from src.data.live_weather import fetch_noaa_metars, metars_to_dataframe

SAMPLE = [
    {"icaoId": "EDDF", "lat": 50.045, "lon": 8.598, "fltCat": "VFR", "visib": "6+", "wdir": "VRB",
     "wspd": 4, "temp": 9, "altim": 1011, "rawOb": "METAR EDDF", "clouds": []},
    {"icaoId": "CWSA", "lat": 45.0, "lon": -73.0, "fltCat": "IFR", "visib": 2, "wdir": 60,
     "wspd": 37, "temp": 9, "altim": 999.1, "rawOb": "SPECI CWSA",
     "clouds": [{"cover": "FEW", "base": 500}, {"cover": "BKN", "base": 1100}, {"cover": "OVC", "base": 1800}]},
    {"icaoId": "BAD", "fltCat": "VFR"},
]


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return SAMPLE


class _FakeSession:
    last_params: dict = {}

    def get(self, url, params=None, timeout=None):
        _FakeSession.last_params = params
        return _FakeResponse()


def test_metars_parse_string_fields_and_skip_bad_records(monkeypatch):
    monkeypatch.setattr(live_weather, "_session", lambda: _FakeSession())
    reports = fetch_noaa_metars()

    assert [r.station_id for r in reports] == ["EDDF", "CWSA"]
    eddf, cwsa = reports
    assert eddf.visibility_mi == 6.0
    assert eddf.wind_dir_deg == 0.0
    assert eddf.flight_category == "VFR"
    assert eddf.ceiling_ft == 99999.0
    assert cwsa.ceiling_ft == 1100.0
    assert abs(cwsa.altimeter_inhg - 29.50) < 0.01


def test_metars_default_to_global_bbox_in_lat_lon_order(monkeypatch):
    monkeypatch.setattr(live_weather, "_session", lambda: _FakeSession())
    fetch_noaa_metars()
    assert _FakeSession.last_params["bbox"] == "-60.0,-180.0,75.0,180.0"

    fetch_noaa_metars(bbox=(20, -100, 60, 30))
    assert _FakeSession.last_params["bbox"] == "20,-100,60,30"


def test_metars_dataframe_has_flight_category(monkeypatch):
    monkeypatch.setattr(live_weather, "_session", lambda: _FakeSession())
    df = metars_to_dataframe(fetch_noaa_metars())
    assert list(df["category"]) == ["VFR", "IFR"]
