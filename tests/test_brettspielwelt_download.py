from io import BytesIO
from urllib.error import HTTPError

import pytest

from scraper.download_brettspielwelt import (
    classify_log,
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
