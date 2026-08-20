from io import BytesIO
from urllib.error import HTTPError

import pytest

from scraper.download_brettspielwelt import (
    classify_log,
    collect_latest,
    discover_latest_id,
    download_game,
    game_ids,
)


COMPLETE_LOG = """---------------Gr.Tichukarten------------------
(0)a Ma R2 G2 B2 S2 R3 G3 B3
---------------Startkarten------------------
(0)a Ma R2 G2 B2 S2 R3 G3 B3 S3 R4 G4 B4 S4 R5
Schupfen:
---------------Rundenverlauf------------------
(0)a: Ma
Ergebnis: 100 - 0
"""


def test_classify_log():
    assert classify_log(COMPLETE_LOG) == "complete"
    assert classify_log(COMPLETE_LOG.replace("Ergebnis: 100 - 0", "")) == "incomplete"
    assert classify_log("server error") == "invalid"


def test_game_ids_uses_latest_count_or_explicit_range():
    assert list(game_ids(latest_id=10, count=3)) == [8, 9, 10]
    assert list(game_ids(latest_id=10, count=3, start_id=4, end_id=5)) == [4, 5]
    with pytest.raises(ValueError):
        game_ids(latest_id=10, count=3, start_id=4)


def test_download_game_writes_valid_log_and_resumes(tmp_path):
    calls = []

    def fetch(url, *, timeout):
        calls.append((url, timeout))
        return COMPLETE_LOG.encode()

    first = download_game(42, output_dir=tmp_path, fetch=fetch)
    second = download_game(42, output_dir=tmp_path, fetch=fetch)

    assert first["status"] == "complete"
    assert second["status"] == "skipped_complete"
    assert (tmp_path / "42.tch").read_text() == COMPLETE_LOG
    assert len(calls) == 1


def test_download_game_does_not_store_invalid_response(tmp_path):
    result = download_game(
        42,
        output_dir=tmp_path,
        fetch=lambda url, timeout: b"internal server error",
    )

    assert result["status"] == "invalid"
    assert not (tmp_path / "42.tch").exists()


def test_download_game_reports_http_error_after_retries(tmp_path):
    calls = []

    def fetch(url, *, timeout):
        calls.append(url)
        raise HTTPError(url, 500, "error", {}, BytesIO())

    result = download_game(42, output_dir=tmp_path, retries=1, fetch=fetch)

    assert result["status"] == "http_500"
    assert len(calls) == 2


def test_discover_latest_id_searches_above_known_valid_hint():
    calls = []

    def fetch(url, *, timeout):
        game_id = int(url.rsplit("/", 1)[-1].removesuffix(".tch"))
        calls.append(game_id)
        if game_id <= 12:
            return COMPLETE_LOG.encode()
        raise HTTPError(url, 500, "not found", {}, BytesIO())

    latest = discover_latest_id(
        hint=8,
        initial_step=2,
        retries=0,
        fetch=fetch,
    )

    assert latest == 12
    assert 8 in calls
    assert 13 in calls


def test_collect_latest_counts_existing_files_and_downloads_newest_first(tmp_path):
    (tmp_path / "8.tch").write_text(COMPLETE_LOG, encoding="utf-8")
    calls = []

    def fetch(url, *, timeout):
        game_id = int(url.rsplit("/", 1)[-1].removesuffix(".tch"))
        calls.append(game_id)
        return COMPLETE_LOG.encode()

    summary = collect_latest(
        latest_id=10,
        target_total=3,
        output_dir=tmp_path,
        delay=0,
        progress_every=1,
        fetch=fetch,
    )

    assert calls == [10, 9]
    assert summary["initial_complete"] == 1
    assert summary["new_complete"] == 2
    assert summary["complete"] == 3
    assert summary["reached_target"]
    assert {path.name for path in tmp_path.glob("*.tch")} == {
        "8.tch",
        "9.tch",
        "10.tch",
    }


def test_collect_latest_continues_past_invalid_ids(tmp_path):
    calls = []

    def fetch(url, *, timeout):
        game_id = int(url.rsplit("/", 1)[-1].removesuffix(".tch"))
        calls.append(game_id)
        if game_id == 9:
            return b"not a tichu log"
        return COMPLETE_LOG.encode()

    summary = collect_latest(
        latest_id=10,
        target_total=2,
        output_dir=tmp_path,
        delay=0,
        progress_every=10,
        fetch=fetch,
    )

    assert calls == [10, 9, 8]
    assert summary["complete"] == 2
    assert summary["counts"] == {"complete": 2, "invalid": 1}
