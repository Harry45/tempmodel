from datetime import date
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import pytest
import requests
import responses

from tempmodel.config import Settings, load_settings
from tempmodel.data.openmeteo import OpenMeteoError, fetch_daily, parse_daily

START, END = date(2026, 1, 1), date(2026, 1, 2)


@pytest.fixture
def settings() -> Settings:
    return load_settings(["api.max_retries=2", "api.retry_backoff_seconds=0"])


def payload(**daily: Any) -> dict[str, Any]:
    values = {
        "time": ["2026-01-01", "2026-01-02"],
        "temperature_2m_mean": [5.0, 6],
        "temperature_2m_max": [8.0, None],
        "temperature_2m_min": [1.0, 2.0],
    }
    return {"daily": values | daily}


def fetch(settings: Settings) -> pd.DataFrame:
    return fetch_daily(settings.api, settings.location, START, END)


@responses.activate
def test_fetches_values_using_the_settings(settings: Settings) -> None:
    responses.get(str(settings.api.url), json=payload())
    data = fetch(settings)
    assert list(data.columns) == list(settings.api.variables)
    assert data.index.name == "date"
    assert data.index.equals(pd.date_range(START, END))
    assert data.loc["2026-01-02", "temperature_2m_mean"] == 6.0
    assert data.isna().sum().sum() == 1

    url = responses.calls[0].request.url
    sent = dict(parse_qsl(urlsplit(str(url)).query))
    assert sent["start_date"] == "2026-01-01"
    assert sent["end_date"] == "2026-01-02"
    assert sent["models"] == "best_match"
    assert sent["timezone"] == "Europe/London"
    assert float(sent["latitude"]) == settings.location.latitude


@responses.activate
def test_http_error_reports_reason(settings: Settings) -> None:
    responses.get(str(settings.api.url), status=400, json={"reason": "bad date"})
    with pytest.raises(OpenMeteoError, match=r"400.*bad date"):
        fetch(settings)


@responses.activate
def test_connection_error_is_wrapped(settings: Settings) -> None:
    responses.get(str(settings.api.url), body=requests.ConnectionError("offline"))
    with pytest.raises(OpenMeteoError, match="offline"):
        fetch(settings)


@responses.activate
def test_response_that_is_not_json_is_rejected(settings: Settings) -> None:
    responses.get(str(settings.api.url), body="<html>")
    with pytest.raises(OpenMeteoError, match="unexpected response"):
        fetch(settings)


@responses.activate
def test_rate_limit_is_retried(settings: Settings) -> None:
    limited = responses.get(str(settings.api.url), status=429)
    responses.get(str(settings.api.url), json=payload())
    assert len(fetch(settings)) == 2
    assert limited.call_count == 1


@responses.activate
def test_gives_up_after_max_retries(settings: Settings) -> None:
    responses.get(str(settings.api.url), status=429, json={"reason": "limit"})
    with pytest.raises(OpenMeteoError, match=r"429.*limit"):
        fetch(settings)
    assert len(responses.calls) == 3


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {},
        {"daily": None},
        payload(time=None),
        payload(time=["not a date", "2026-01-02"]),
        payload(temperature_2m_max=None),
        payload(temperature_2m_max=[1.0]),
        payload(temperature_2m_max=["warm", "cold"]),
    ],
)
def test_unusable_payload_is_rejected(bad: object) -> None:
    with pytest.raises(OpenMeteoError, match="unexpected response"):
        parse_daily(bad, ["temperature_2m_mean", "temperature_2m_max"])
