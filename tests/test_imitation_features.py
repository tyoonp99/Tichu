import json

import numpy as np
import pytest

from nn_training.imitation_features import (
    FeatureEncoder,
    FeatureSchema,
    buffered_shuffle,
    iter_jsonl,
)


def sample_record():
    pass_action = {"type": "PassAction", "player": 0}
    play_action = {
        "type": "PlayCombination",
        "player": 0,
        "combination": "Single",
        "cards": ["A_JADE"],
        "height": 14,
    }
    return {
        "observation": {
            "hand": ["A_JADE", "PHOENIX"],
            "hand_sizes": [2, 5, 4, 3],
            "won_trick_counts": [1, 0, 2, 1],
            "won_trick_points": [15, 0, 25, 10],
            "trick": [{"type": "PassAction", "player": 3}],
            "wish": None,
            "ranking": [2],
            "announced_tichu": [1],
            "announced_grand_tichu": [],
        },
        "legal_actions": [pass_action, play_action],
        "chosen_action": play_action,
    }


def test_schema_and_encoder_keep_chosen_action_index():
    record = sample_record()
    schema = FeatureSchema.from_records([record], max_trick_actions=3)
    encoder = FeatureEncoder(schema)

    state, candidates, chosen_index = encoder.encode_record(record)

    assert state.shape == (encoder.state_size,)
    assert candidates.shape == (2, encoder.action_size)
    assert chosen_index == 1
    assert np.isfinite(state).all()
    assert np.isfinite(candidates).all()


def test_trick_history_is_right_aligned_and_order_sensitive():
    record = sample_record()
    schema = FeatureSchema.from_records([record], max_trick_actions=2)
    encoder = FeatureEncoder(schema)
    one_action = encoder.encode_observation(record["observation"])

    changed = sample_record()
    changed["observation"]["trick"].insert(
        0,
        {
            "type": "PlayCombination",
            "player": 2,
            "combination": "Single",
            "cards": ["A_JADE"],
            "height": 14,
        },
    )
    two_actions = encoder.encode_observation(changed["observation"])

    action_size = encoder.action_size
    assert np.count_nonzero(one_action[-2 * action_size : -action_size]) == 0
    assert not np.array_equal(one_action, two_actions)


def test_encoder_rejects_chosen_action_outside_legal_actions():
    record = sample_record()
    record["chosen_action"] = {"type": "PassAction", "player": 1}
    encoder = FeatureEncoder(FeatureSchema.from_records([record]))

    with pytest.raises(ValueError, match="absent"):
        encoder.encode_record(record)


def test_jsonl_reader_reports_bad_line(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(sample_record()) + "\nnot-json\n", encoding="utf-8")

    records = iter_jsonl(path)
    assert next(records)["chosen_action"]["type"] == "PlayCombination"
    with pytest.raises(ValueError, match=":2"):
        next(records)


def test_buffered_shuffle_preserves_all_records():
    records = [{"id": index} for index in range(20)]
    shuffled = list(
        buffered_shuffle(records, buffer_size=5, rng=np.random.default_rng(42))
    )

    assert sorted(record["id"] for record in shuffled) == list(range(20))
    assert shuffled != records
