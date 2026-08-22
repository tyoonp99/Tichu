"""Privacy-safe training records for Tichu agent decisions."""

import hashlib
import json
from pathlib import Path

from gym_tichu.envs.internals import (
    GiveDragonAwayAction,
    PassAction,
    PassBombAction,
    PlayCombination,
    TichuAction,
    WinTrickAction,
    WishAction,
)


SCHEMA_VERSION = 2


def _relative_position(position, observer):
    return (position - observer) % 4


def _card_names(cards):
    return [card.name for card in sorted(cards, key=lambda card: card.number)]


def encode_action(action, observer):
    """Encode an action using positions relative to the acting player."""
    encoded = {
        "type": action.__class__.__name__,
        "player": _relative_position(action.player_pos, observer),
    }
    if isinstance(action, PlayCombination):
        encoded.update(
            {
                "combination": action.combination.__class__.__name__,
                "cards": _card_names(action.combination.cards),
                "height": action.combination.height,
            }
        )
    elif isinstance(action, TichuAction):
        encoded.update({"announce": action.announce, "grand": action.grand})
    elif isinstance(action, WishAction):
        encoded["wish"] = action.wish.name if action.wish else None
    elif isinstance(action, GiveDragonAwayAction):
        encoded["to"] = _relative_position(action.to, observer)
        encoded["trick_points"] = action.trick.points
    elif isinstance(action, WinTrickAction):
        encoded["trick_points"] = action.trick.points
    elif not isinstance(action, (PassAction, PassBombAction)):
        encoded["display"] = str(action)
    return encoded


def encode_observation(state, observer):
    """Encode only information available to ``observer`` during play.

    Opponent and partner cards, the complete state object, and state history are
    deliberately excluded. Player-indexed public features are rotated so index
    zero always means the acting player, followed by right opponent, partner,
    and left opponent.
    """
    positions = [(observer + offset) % 4 for offset in range(4)]
    bomb_resume_player = state.bomb_resume_player
    return {
        "hand": _card_names(state.handcards[observer]),
        "hand_sizes": [len(state.handcards[position]) for position in positions],
        "won_trick_counts": [len(state.won_tricks[position]) for position in positions],
        "won_trick_points": [
            sum(trick.points for trick in state.won_tricks[position])
            for position in positions
        ],
        "trick": [encode_action(action, observer) for action in state.trick_on_table],
        "wish": state.wish.name if state.wish else None,
        "ranking": [_relative_position(position, observer) for position in state.ranking],
        "announced_tichu": sorted(
            _relative_position(position, observer) for position in state.announced_tichu
        ),
        "announced_grand_tichu": sorted(
            _relative_position(position, observer)
            for position in state.announced_grand_tichu
        ),
        # Do not serialize the complete bomb_window: it is derived from every
        # player's hidden hand and would leak who else owns a bomb.  These
        # fields contain only the public decision context needed by the actor.
        "decision_context": "bomb_response" if state.bomb_window else "normal",
        "bomb_resume_player": (
            _relative_position(bomb_resume_player, observer)
            if bomb_resume_player is not None
            else None
        ),
        "bomb_trick_finish": bool(state.bomb_trick_finish),
    }


def extract_search_statistics(agent, state, observer):
    """Return root-action MCTS statistics when the wrapped agent exposes them."""
    search = getattr(agent, "_search", None)
    graph = getattr(search, "graph", None)
    if search is None or graph is None or len(state.possible_actions_set) <= 1:
        return []

    try:
        root_id = search.graph_node_id(state)
        edges = graph.out_edges(root_id, data="action", default=None)
    except (KeyError, TypeError):
        return []

    statistics = []
    legal_actions = state.possible_actions_set
    for _, child_id, action in edges:
        if action not in legal_actions:
            continue
        record = graph.nodes[child_id]["record"]
        mean_reward = (
            record.total_reward[observer] / record.visit_count
            if record.visit_count
            else None
        )
        statistics.append(
            {
                "action": encode_action(action, observer),
                "visits": record.visit_count,
                "availability": record.availability_count,
                "mean_reward": mean_reward,
            }
        )
    return sorted(statistics, key=lambda item: json.dumps(item["action"], sort_keys=True))


class DecisionRecorder:
    """Collect a game's decisions and attach its final team result."""

    def __init__(self, *, game_id, seed, swapped):
        self.game_id = game_id
        self.seed = seed
        self.swapped = swapped
        self.records = []

    def record(self, *, state, action, agent):
        observer = state.player_pos
        self.records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "game_id": self.game_id,
                "seed": self.seed,
                "swapped": self.swapped,
                "decision": len(self.records),
                "player": observer,
                "agent": agent.__class__.__name__,
                "observation": encode_observation(state, observer),
                "legal_actions": [
                    encode_action(legal_action, observer)
                    for legal_action in sorted(state.possible_actions_list, key=repr)
                ],
                "chosen_action": encode_action(action, observer),
                "search_statistics": extract_search_statistics(agent, state, observer),
                "outcome": None,
            }
        )

    def finalize(self, points):
        even_points, odd_points = points
        for record in self.records:
            player = record["player"]
            team_points, opponent_points = (
                (even_points, odd_points) if player % 2 == 0 else (odd_points, even_points)
            )
            record["outcome"] = {
                "team_points": team_points,
                "opponent_points": opponent_points,
                "point_diff": team_points - opponent_points,
                "won": team_points > opponent_points,
            }
        return self.records


class RecordingAgent:
    """Transparent agent wrapper that records play/pass decisions."""

    def __init__(self, agent, recorder):
        self.agent = agent
        self.recorder = recorder

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def action(self, state):
        action = self.agent.action(state)
        if action not in state.possible_actions_set:
            raise ValueError("Agent selected an illegal action: {}".format(action))
        self.recorder.record(state=state, action=action, agent=self.agent)
        return action


def split_for_seed(seed, validation_fraction=0.1):
    """Assign a seed deterministically so paired games never cross splits."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    digest = hashlib.sha256(str(seed).encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if bucket < validation_fraction else "train"


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")
