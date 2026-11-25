import json
import logging
import sys
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
import responses

from tempmodel.config import load_settings
from tempmodel.data import load_csv
from tempmodel.data.download import main

URL = str(load_settings().api.url)

Run = Callable[..., None]


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Run]:
    """Run the command, keeping its files and logging settings to this test."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]

    def run_command(*overrides: str) -> None:
        paths = [f"output.path={tmp_path}/data.csv", f"hydra.run.dir={tmp_path}/run"]
        arguments = ["tempmodel-download", *paths, "api.max_retries=0", *overrides]
        monkeypatch.setattr(sys, "argv", arguments)
        main()

    yield run_command
    for handler in root.handlers[:]:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def reply(request: Any) -> tuple[int, dict[str, str], str]:
    """Answer as Open-Meteo does, with a value for every day requested."""
    start = date.fromisoformat(request.params["start_date"])
    end = date.fromisoformat(request.params["end_date"])
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    daily = {
        "time": [day.isoformat() for day in days],
        "temperature_2m_mean": [5.0] * len(days),
        "temperature_2m_max": [8.0] * len(days),
        "temperature_2m_min": [2.0] * len(days),
    }
    return 200, {}, json.dumps({"daily": daily})


def log_of(tmp_path: Path) -> str:
    return (tmp_path / "run" / "download.log").read_text()


@responses.activate
def test_saves_the_data_and_a_log(run: Run, tmp_path: Path) -> None:
    responses.add_callback(responses.GET, URL, callback=reply)
    run("period.years=1")
    data = load_csv(tmp_path / "data.csv")
    assert len(data) in (365, 366)
    assert list(data.columns) == list(load_settings().api.variables)
    assert data.index.equals(pd.date_range(data.index[0], data.index[-1]))
    assert f"Saved {len(data)} days" in log_of(tmp_path)
    assert (tmp_path / "run" / ".hydra" / "config.yaml").exists()


@responses.activate
def test_invalid_setting_stops_before_any_request(run: Run, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stopped:
        run("period.years=0")
    assert stopped.value.code == 1
    assert "Invalid configuration" in log_of(tmp_path)
    assert not responses.calls


@pytest.mark.parametrize(
    "answer",
    [
        {"status": 400, "json": {"reason": "bad model"}},
        {"body": requests.ConnectionError("offline")},
        {"json": {"daily": {"time": []}}},
    ],
)
@responses.activate
def test_failure_keeps_the_existing_file(
    run: Run, tmp_path: Path, answer: dict[str, Any]
) -> None:
    (tmp_path / "data.csv").write_text("old")
    responses.get(URL, **answer)
    with pytest.raises(SystemExit) as stopped:
        run()
    assert stopped.value.code == 1
    assert "Download failed" in log_of(tmp_path)
    assert (tmp_path / "data.csv").read_text() == "old"
