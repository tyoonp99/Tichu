import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from nn_training.model_c import ModelCEncoder, ModelCPolicy
from nn_training.imitation_features import FeatureSchema


def _record(index):
    pass_action = {"type": "PassAction", "player": 0}
    play_action = {"type": "PlayCombination", "player": 0, "combination": "Single", "cards": ["A_JADE"], "height": 14}
    return {
        "schema_version": 2,
        "observation": {
            "hand": ["A_JADE", "PHOENIX"], "hand_sizes": [2, 3, 4, 5],
            "won_trick_counts": [0, 1, 0, 0], "won_trick_points": [0, 5, 0, 0],
            "trick": [play_action], "wish": None, "ranking": [],
            "announced_tichu": [], "announced_grand_tichu": [],
            "decision_context": "bomb_response" if index else "normal",
            "bomb_resume_player": None, "bomb_trick_finish": False,
        },
        "legal_actions": [pass_action, play_action],
        "chosen_action": play_action if index else pass_action,
    }


def test_model_c_encodes_tokens_and_masks_illegal_candidates():
    records = [_record(0), _record(1)]
    schema = FeatureSchema.from_records(records)
    encoder = ModelCEncoder(schema)
    batch = encoder.encode_batch(records)
    model = ModelCPolicy(**encoder.model_token_config(), width=32, heads=4, dropout=0)

    logits = model(batch)

    assert batch.hand_cards.shape == (2, 14)
    assert batch.trick_type.shape == (2, schema.max_trick_actions)
    assert batch.candidate_cards.shape == (2, 2, 14)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits[batch.legal_action_mask]).all()
    weighted = ModelCEncoder(schema, target_weights={"Single": 3.0}).encode_batch(
        records
    )
    assert weighted.target_weights.tolist() == [1.0, 3.0]

    one_action = encoder.encode_batch([{
        **records[0], "legal_actions": records[0]["legal_actions"][:1], "chosen_action": records[0]["legal_actions"][0]
    }])
    padded = replace(
        one_action,
        candidate_type=torch.cat((one_action.candidate_type, torch.zeros((1, 1), dtype=torch.long)), dim=1),
        candidate_combination=torch.cat((one_action.candidate_combination, torch.zeros((1, 1), dtype=torch.long)), dim=1),
        candidate_player=torch.cat((one_action.candidate_player, torch.zeros((1, 1), dtype=torch.long)), dim=1),
        candidate_cards=torch.cat((one_action.candidate_cards, torch.zeros((1, 1, 14), dtype=torch.long)), dim=1),
        candidate_card_mask=torch.cat((one_action.candidate_card_mask, torch.zeros((1, 1, 14), dtype=torch.bool)), dim=1),
        candidate_numeric=torch.cat(
            (one_action.candidate_numeric, torch.zeros((1, 1, 5))), dim=1
        ),
        legal_action_mask=torch.tensor([[True, False]]),
    )
    assert math.isinf(model(padded)[0, 1].item())
