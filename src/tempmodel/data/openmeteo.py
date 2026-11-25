"""Fetch daily data from the Open-Meteo archive."""

import logging
from collections.abc import Sequence
from datetime import date

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from tempmodel.config import ApiSettings, LocationSettings

logger = logging.getLogger(__name__)

RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_ERROR_LENGTH = 300


class OpenMeteoError(RuntimeError):
    """The API could not be reached or returned something unusable."""


def fetch_daily(
    api: ApiSettings, location: LocationSettings, start: date, end: date
) -> pd.DataFrame:
    """Fetch daily values from ``start`` to ``end`` inclusive, indexed by local date.

    Raises:
        OpenMeteoError: If the request fails or the response is not usable.
    """
    logger.info("Requesting %s for %s, %s to %s", api.model, location.name, start, end)
    payload = request_daily(api, location, start, end)
    return parse_daily(payload, api.variables)


def request_daily(
    api: ApiSettings, location: LocationSettings, start: date, end: date
) -> object:
    """Return the decoded JSON response, retrying temporary failures."""
    params: dict[str, str | float] = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(api.variables),
        "timezone": location.timezone,
        "models": api.model,
    }
    try:
        with _retrying_session(api) as session:
            response = session.get(
                str(api.url), params=params, timeout=api.timeout_seconds
            )
    except requests.RequestException as error:
        raise OpenMeteoError(f"request failed: {error}") from error

    if not response.ok:
        raise OpenMeteoError(
            f"HTTP {response.status_code}: {response.text[:MAX_ERROR_LENGTH]}"
        )
    try:
        payload: object = response.json()
    except requests.JSONDecodeError as error:
        raise OpenMeteoError(f"unexpected response: {error}") from error
    return payload


def parse_daily(payload: object, variables: Sequence[str]) -> pd.DataFrame:
    """Return the daily values in a response as a table."""
    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict):
        raise OpenMeteoError("unexpected response: no daily data")

    missing = [
        name for name in ("time", *variables) if not isinstance(daily.get(name), list)
    ]
    if missing:
        raise OpenMeteoError(f"unexpected response: missing {missing}")

    # pandas would repeat a single value to fill a column, so check explicitly.
    if any(len(daily[name]) != len(daily["time"]) for name in variables):
        raise OpenMeteoError("unexpected response: series have different lengths")

    try:
        dates = pd.to_datetime(pd.Index(daily["time"], name="date"), format="%Y-%m-%d")
        values = {name: daily[name] for name in variables}
        return pd.DataFrame(values, index=dates, dtype=float)
    except (TypeError, ValueError) as error:  # not dates or not numbers
        raise OpenMeteoError(f"unexpected response: {error}") from error


def _retrying_session(api: ApiSettings) -> requests.Session:
    retry = Retry(
        total=api.max_retries,
        status_forcelist=RETRY_STATUSES,
        backoff_factor=api.retry_backoff_seconds,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
