import gzip
import json

import pytest

from nn_training.decision_dataset import (
    JsonlDecisionDataset,
    collate_decision_records,
    make_decision_dataloader,
    make_tensor_dataloader,
)
from nn_training.imitation_features import FeatureEncoder, FeatureSchema


def _record(index):
    return {
        "id": index,
        "observation": {
            "hand": ["A_JADE"],
            "hand_sizes": [1, 2, 3, 4],
            "won_trick_counts": [0, 0, 0, 0],
            "won_trick_points": [0, 0, 0, 0],
            "trick": [],
            "wish": None,
            "ranking": [],
            "announced_tichu": [],
            "announced_grand_tichu": [],
        },
        "legal_actions": [{"type": "PassAction", "player": 0}],
        "chosen_action": {"type": "PassAction", "player": 0},
    }


def _gzip_jsonl(tmp_path, count=5):
    path = tmp_path / "decisions.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for index in range(count):
            output.write(json.dumps(_record(index)) + "\n")
    return path


def test_dataset_streams_gzip_jsonl_and_honors_max_records(tmp_path):
    path = _gzip_jsonl(tmp_path)

    records = list(JsonlDecisionDataset(path, max_records=3))

    assert [record["id"] for record in records] == [0, 1, 2]


def test_dataloader_batches_variable_length_records_without_default_collation(tmp_path):
    path = _gzip_jsonl(tmp_path)
    loader = make_decision_dataloader(path, batch_size=2)

    batches = list(loader)

    assert [[record["id"] for record in batch] for batch in batches] == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_dataset_shuffle_is_seeded_and_preserves_records(tmp_path):
    path = _gzip_jsonl(tmp_path, count=20)

    first = [
        record["id"]
        for record in JsonlDecisionDataset(path, shuffle_buffer=5, seed=42)
    ]
    second = [
        record["id"]
        for record in JsonlDecisionDataset(path, shuffle_buffer=5, seed=42)
    ]

    assert first == second
    assert sorted(first) == list(range(20))
    assert first != list(range(20))


def test_dataloader_workers_partition_records_without_duplicates(tmp_path):
    path = _gzip_jsonl(tmp_path, count=12)
    loader = make_decision_dataloader(path, batch_size=2, num_workers=2)

    ids = [record["id"] for batch in loader for record in batch]

    assert sorted(ids) == list(range(12))


def test_tensor_dataloader_yields_padded_candidate_batch(tmp_path):
    path = _gzip_jsonl(tmp_path, count=3)
    records = list(JsonlDecisionDataset(path))
    encoder = FeatureEncoder(FeatureSchema.from_records(records))
    loader = make_tensor_dataloader(path, encoder=encoder, batch_size=2)

    batch = next(iter(loader))

    assert batch.states.shape == (2, encoder.state_size)
    assert batch.actions.shape == (2, 1, encoder.action_size)
    assert batch.legal_action_mask.tolist() == [[True], [True]]
    assert batch.targets.tolist() == [0, 0]


@pytest.mark.parametrize(
    "factory,expected",
    [
        (lambda path: JsonlDecisionDataset(path, max_records=0), "max_records"),
        (lambda path: JsonlDecisionDataset(path, shuffle_buffer=0), "shuffle_buffer"),
        (lambda path: make_decision_dataloader(path, batch_size=0), "batch_size"),
        (
            lambda path: make_decision_dataloader(
                path, batch_size=1, num_workers=-1
            ),
            "num_workers",
        ),
    ],
)
def test_dataset_rejects_invalid_loader_options(tmp_path, factory, expected):
    path = _gzip_jsonl(tmp_path)

    with pytest.raises(ValueError, match=expected):
        factory(path)


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty"):
        collate_decision_records([])
