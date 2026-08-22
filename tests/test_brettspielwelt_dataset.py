import json

import pytest

from scraper.prepare_brettspielwelt_dataset import (
    anonymized_game_id,
    _open_text_output,
    _sha256_file,
    split_for_game,
    validate_split_manifest,
    winning_team,
)


def test_game_split_is_deterministic():
    assert split_for_game(2419834, 0.1) == split_for_game(2419834, 0.1)


def test_game_split_supports_disjoint_train_validation_and_test_sets():
    assignments = {
        game_id: split_for_game(game_id, validation_fraction=0.2, test_fraction=0.2)
        for game_id in range(1000, 1100)
    }

    assert set(assignments.values()) == {"train", "validation", "test"}
    assert len(assignments) == 100


@pytest.mark.parametrize(
    "validation_fraction,test_fraction",
    [(-0.1, 0.1), (0.1, -0.1), (0.5, 0.5), (0.8, 0.3)],
)
def test_game_split_rejects_invalid_fractions(
    validation_fraction, test_fraction
):
    with pytest.raises(ValueError):
        split_for_game(
            2419834,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )


def test_public_game_id_is_stable_and_hides_download_id():
    public_id = anonymized_game_id(2419834)

    assert public_id == anonymized_game_id(2419834)
    assert public_id.startswith("bsw-")
    assert "2419834" not in public_id


def test_deterministic_gzip_output_has_stable_hash(tmp_path):
    paths = [tmp_path / "first.jsonl.gz", tmp_path / "second.jsonl.gz"]
    for path in paths:
        with _open_text_output(path, compress=True) as output:
            output.write('{"value": 1}\n')

    assert _sha256_file(paths[0]) == _sha256_file(paths[1])


def test_split_manifest_detects_cross_split_game_leakage(tmp_path):
    path = tmp_path / "split-manifest.jsonl"
    rows = [
        {"game_id": "bsw-a", "split": "train"},
        {"game_id": "bsw-b", "split": "validation"},
        {"game_id": "bsw-a", "split": "test"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = validate_split_manifest(path)

    assert result["passed"] is False
    assert result["game_ids_in_multiple_splits"] == 1


def test_winning_team_uses_even_and_odd_seat_teams():
    assert winning_team((120, 80)) == 0
    assert winning_team((45, 155)) == 1
    assert winning_team((100, 100)) is None
