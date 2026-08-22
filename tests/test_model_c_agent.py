import pytest

torch = pytest.importorskip("torch")

from benchmark import make_agent  # noqa: E402
from gym_agents import ModelCAgent, ModelCGuidedMctsAgent  # noqa: E402
from gym_agents.training_data import encode_action, encode_observation  # noqa: E402
from gym_tichu.envs.internals import Card  # noqa: E402
from nn_training.imitation_features import FeatureSchema  # noqa: E402
from nn_training.model_c import ModelCEncoder, ModelCPolicy  # noqa: E402


def make_model_c_checkpoint(path, state):
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
    encoder = ModelCEncoder(schema)
    model = ModelCPolicy(**encoder.model_token_config(), width=32, heads=4, dropout=0)
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


def test_model_c_agent_returns_only_a_legal_engine_action(tmp_path, state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=0,
    )
    checkpoint = tmp_path / "model-c.pt"
    make_model_c_checkpoint(checkpoint, state)

    agent = ModelCAgent(checkpoint, device="cpu")
    chosen = agent.action(state)

    assert chosen in state.possible_actions_set
    assert len(agent.last_scored_actions) == len(state.possible_actions_list)


def test_benchmark_factory_creates_model_c_agent(tmp_path, state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    checkpoint = tmp_path / "model-c.pt"
    make_model_c_checkpoint(checkpoint, state)

    agent = make_agent(
        "model-c", model_c_checkpoint=checkpoint, model_c_device="cpu"
    )

    assert isinstance(agent, ModelCAgent)


def test_benchmark_factory_creates_model_c_v1_agent(tmp_path, state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    checkpoint = tmp_path / "model-c-v1.pt"
    make_model_c_checkpoint(checkpoint, state)

    agent = make_agent(
        "model-c-v1", model_c_v1_checkpoint=checkpoint, model_c_device="cpu"
    )

    assert isinstance(agent, ModelCAgent)


def test_model_c_guided_mcts_uses_legal_actions_and_policy_prior(tmp_path, state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    checkpoint = tmp_path / "model-c.pt"
    make_model_c_checkpoint(checkpoint, state)

    agent = ModelCGuidedMctsAgent(
        checkpoint, iterations=1, max_time=1, cheat=True, device="cpu"
    )
    chosen = agent.action(state)

    assert chosen in state.possible_actions_set
    assert len(agent.policy.last_scored_actions) == len(state.possible_actions_list)
