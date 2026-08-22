import copy
import math

import pytest

torch = pytest.importorskip("torch")

from gym_agents.model_c_selfplay import ModelCSelfPlayAgent, SelfPlayTrajectory  # noqa: E402
from gym_agents.model_c_rl import ModelCPPOAgent  # noqa: E402
from gym_agents.training_data import encode_action, encode_observation  # noqa: E402
from gym_tichu.envs.internals import Card  # noqa: E402
from nn_training.imitation_features import FeatureSchema  # noqa: E402
from nn_training.model_c import ModelCEncoder, ModelCPolicy  # noqa: E402
from nn_training.selfplay import (  # noqa: E402
    ModelCActorCritic,
    SelfPlayTransition,
    compute_gae,
    ppo_update,
    save_selfplay_checkpoint,
)


def records():
    passed = {"type": "PassAction", "player": 0}
    played = {
        "type": "PlayCombination",
        "player": 0,
        "combination": "Single",
        "cards": ["A_JADE"],
        "height": 14,
    }
    observation = {
        "hand": ["A_JADE", "PHOENIX"],
        "hand_sizes": [2, 3, 4, 5],
        "won_trick_counts": [0, 1, 0, 0],
        "won_trick_points": [0, 5, 0, 0],
        "trick": [played],
        "wish": None,
        "ranking": [],
        "announced_tichu": [],
        "announced_grand_tichu": [],
        "decision_context": "normal",
        "bomb_resume_player": None,
        "bomb_trick_finish": False,
    }
    return [
        {"schema_version": 2, "observation": observation, "legal_actions": [passed, played], "chosen_action": passed},
        {"schema_version": 2, "observation": observation, "legal_actions": [passed, played], "chosen_action": played},
    ]


def make_actor_critic():
    schema = FeatureSchema.from_records(records())
    encoder = ModelCEncoder(schema)
    policy = ModelCPolicy(
        **encoder.model_token_config(), width=32, heads=4, dropout=0
    )
    return ModelCActorCritic(policy), encoder


def test_actor_critic_preserves_legal_action_mask_and_adds_one_value():
    actor_critic, encoder = make_actor_critic()
    batch = encoder.encode_batch(records())

    logits, values = actor_critic(batch)

    assert logits.shape == (2, 2)
    assert values.shape == (2,)
    assert torch.isfinite(logits[batch.legal_action_mask]).all()


def test_ppo_update_changes_actor_critic_on_sampled_legal_actions():
    actor_critic, encoder = make_actor_critic()
    batch = encoder.encode_batch(records())
    with torch.no_grad():
        logits, values = actor_critic(batch)
        old_log_probs = torch.log_softmax(logits, dim=1).gather(
            1, batch.targets[:, None]
        ).squeeze(1)
    transitions = [
        SelfPlayTransition(record, 0, float(old_log_prob), float(value), reward)
        for record, old_log_prob, value, reward in zip(
            records(), old_log_probs, values, (1.0, -1.0)
        )
    ]
    reference = copy.deepcopy(actor_critic.policy)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    before = actor_critic.value_head[-1].weight.detach().clone()

    metrics = ppo_update(
        actor_critic=actor_critic,
        reference_policy=reference,
        encoder=encoder,
        transitions=transitions,
        optimizer=optimizer,
        device="cpu",
        ppo_epochs=1,
        minibatch_size=1,
    )

    assert metrics["transitions"] == 2
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["kl"])
    assert metrics["kl"] >= 0
    assert not torch.equal(before, actor_critic.value_head[-1].weight)
    assert all(module.p == 0 for module in actor_critic.modules() if isinstance(module, torch.nn.Dropout))


def test_round_rewards_use_the_player_team_perspective():
    trajectory = SelfPlayTrajectory()
    record = records()[0]
    trajectory.append(SelfPlayTransition(record, 0, 0.0, 0.0, 0.0))
    trajectory.append(SelfPlayTransition(record, 1, 0.0, 0.0, 0.0))

    trajectory.reward_round(0, (80, 20))

    assert trajectory.transitions[0].reward == 0
    assert trajectory.transitions[0].done is False
    assert trajectory.transitions[1].reward == pytest.approx(-0.6)
    assert trajectory.transitions[1].done is True


def test_gae_propagates_a_round_reward_back_to_earlier_team_actions():
    record = records()[0]
    transitions = [
        SelfPlayTransition(record, 0, 0.0, 0.2, 0.0),
        SelfPlayTransition(record, 0, 0.0, 0.4, 1.0, done=True),
    ]

    completed = compute_gae(transitions, gamma=1.0, gae_lambda=1.0)

    assert completed[1].advantage == pytest.approx(0.6)
    assert completed[0].advantage == pytest.approx(0.8)
    assert completed[0].return_ == pytest.approx(1.0)
    assert completed[1].return_ == pytest.approx(1.0)


def test_selfplay_agent_samples_only_a_legal_engine_action(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=0,
    )
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
    actor_critic = ModelCActorCritic(
        ModelCPolicy(**encoder.model_token_config(), width=32, heads=4, dropout=0)
    )
    trajectory = SelfPlayTrajectory()
    agent = ModelCSelfPlayAgent(actor_critic, encoder, trajectory, device="cpu")

    chosen = agent.action(state)

    assert chosen in state.possible_actions_set
    assert len(trajectory.transitions) == 1


def test_ppo_agent_loads_checkpoint_and_returns_a_legal_engine_action(tmp_path, state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=0,
    )
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
    actor_critic = ModelCActorCritic(
        ModelCPolicy(**encoder.model_token_config(), width=32, heads=4, dropout=0)
    )
    checkpoint = tmp_path / "selfplay.pt"
    save_selfplay_checkpoint(
        checkpoint,
        actor_critic=actor_critic,
        feature_schema=schema,
        base_checkpoint="model-c.pt",
        update=1,
        metrics={},
    )

    agent = ModelCPPOAgent(checkpoint, device="cpu")
    chosen = agent.action(state)

    assert chosen in state.possible_actions_set
