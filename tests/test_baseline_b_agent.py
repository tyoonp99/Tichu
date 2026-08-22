import pytest

torch = pytest.importorskip("torch")

from benchmark import make_agent  # noqa: E402
from gym_agents import BaselineBAgent  # noqa: E402
from gym_agents.training_data import encode_action, encode_observation  # noqa: E402
from gym_tichu.envs.internals import Card  # noqa: E402
from nn_training.behavior_cloning import BaselineBPolicy  # noqa: E402
from nn_training.imitation_features import FeatureEncoder, FeatureSchema  # noqa: E402


def make_baseline_b_checkpoint(path, state):
    observer = state.player_pos
    legal_actions = [
        encode_action(action, observer)
        for action in sorted(state.possible_actions_list, key=repr)
    ]
    record = {
        "schema_version": 2,
        "observation": encode_observation(state, observer),
        "legal_actions": legal_actions,
        "chosen_action": legal_actions[0],
    }
    schema = FeatureSchema.from_records([record])
    encoder = FeatureEncoder(schema)
    model = BaselineBPolicy(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        hidden_size=8,
        dropout=0,
    )
    torch.save(
        {
            "format_version": 2,
            "model": model.checkpoint_name,
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "feature_schema": schema.to_dict(),
            "epoch": 1,
            "validation": {},
        },
        path,
    )


def test_baseline_b_agent_returns_only_a_legal_engine_action(tmp_path, state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=0,
    )
    checkpoint = tmp_path / "baseline-b.pt"
    make_baseline_b_checkpoint(checkpoint, state)

    agent = BaselineBAgent(checkpoint, device="cpu")
    chosen = agent.action(state)

    assert chosen in state.possible_actions_set
    assert len(agent.last_scored_actions) == len(state.possible_actions_list)


def test_benchmark_factory_creates_baseline_b_agent(tmp_path, state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    checkpoint = tmp_path / "baseline-b.pt"
    make_baseline_b_checkpoint(checkpoint, state)

    agent = make_agent(
        "baseline-b",
        baseline_b_checkpoint=checkpoint,
        baseline_b_device="cpu",
    )

    assert isinstance(agent, BaselineBAgent)
