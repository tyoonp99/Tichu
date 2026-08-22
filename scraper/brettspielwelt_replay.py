"""Replay parsed Brettspielwelt rounds with the local Tichu rules engine."""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from gym_tichu.envs.internals.cards import Card, CardRank, Combination, FullHouse, Straight
from gym_tichu.envs.internals.actions import (
    GiveDragonAwayAction,
    PassAction,
    PassBombAction,
    PlayBomb,
    PlayCombination,
    Trick,
    WinTrickAction,
    WishAction,
)
from gym_tichu.envs.internals.tichu_state import (
    HandCards,
    History,
    TichuState,
    WonTricks,
)

from scraper.brettspielwelt_parser import final_hands, parse_file, validate_game


SPECIAL_CARDS = {
    "Ph": Card.PHOENIX,
    "Dr": Card.DRAGON,
    "Ma": Card.MAHJONG,
    "Hu": Card.DOG,
}
SUIT_NAMES = {
    "R": "HOUSE",  # red stars; the engine retains the historical HOUSE name
    "G": "JADE",
    "B": "PAGODA",
    "S": "SWORD",
}
RANK_NAMES = {
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
    "10": "TEN",
    "B": "J",
    "D": "Q",
    "K": "K",
    "A": "A",
}
WISH_RANKS = {
    token: getattr(CardRank, rank_name) for token, rank_name in RANK_NAMES.items()
}


@dataclass
class ReplayResult:
    status: str
    decisions: list[tuple[object, object, object]] = field(default_factory=list)
    event_index: int | None = None
    detail: str | None = None
    expected_scores: tuple[int, int] | None = None
    actual_scores: tuple[int, int] | None = None
    final_state: object | None = None


def card_from_token(token):
    if token in SPECIAL_CARDS:
        return SPECIAL_CARDS[token]
    suit_name = SUIT_NAMES[token[0]]
    rank_name = RANK_NAMES[token[1:]]
    return getattr(Card, "{}_{}".format(rank_name, suit_name))


def make_initial_state(round_):
    hands = final_hands(round_)
    engine_hands = [frozenset(card_from_token(token) for token in hands[player]) for player in range(4)]
    mahjong_players = [player for player, hand in enumerate(engine_hands) if Card.MAHJONG in hand]
    if len(mahjong_players) != 1:
        raise ValueError("expected exactly one Mahjong holder")
    return TichuState(
        player_pos=mahjong_players[0],
        handcards=HandCards(*engine_hands),
        won_tricks=WonTricks(),
        trick_on_table=Trick(),
        wish=None,
        ranking=(),
        announced_tichu=frozenset(round_.tichu),
        announced_grand_tichu=frozenset(round_.grand_tichu),
        history=History(),
        allow_tichu=False,
        allow_wish=True,
    )


def _action_kind(actions):
    if not actions:
        return "none"
    if all(isinstance(action, WishAction) for action in actions):
        return "wish"
    if all(isinstance(action, GiveDragonAwayAction) for action in actions):
        return "dragon"
    if all(isinstance(action, WinTrickAction) for action in actions):
        return "win_trick"
    if any(isinstance(action, PassBombAction) for action in actions):
        return "bomb"
    return "play"


def _prepare_for_event(state, event=None, inferred_decisions=None):
    """Apply engine-only administrative actions omitted from text logs."""
    event_kind = event.kind if event is not None else "end"
    while not state.is_terminal():
        actions = state.possible_actions_list
        kind = _action_kind(actions)
        if kind == "win_trick":
            if (
                event is not None
                and event.kind == "pass"
                and actions[0].player_pos == event.player
            ):
                break
            state = state.next_state(actions[0])
            continue
        if kind == "wish" and event_kind != "wish":
            no_wish = next(action for action in actions if action.wish is None)
            state = state.next_state(no_wish)
            continue
        if kind == "bomb":
            # BSW only records a bomb when it is played, not each player's
            # implicit decision to refrain from bombing.  Preserve a matching
            # logged bomb and auto-advance all omitted declines.
            if event is not None and event.kind == "play":
                matching, _ = _matching_action(state, event)
                if isinstance(matching, PlayBomb):
                    break
            decline = next(
                action for action in actions if isinstance(action, PassBombAction)
            )
            if inferred_decisions is not None:
                inferred_decisions.append((state, None, decline))
            state = state.next_state(decline)
            continue
        break
    return state


def _matching_action(state, event):
    actions = state.possible_actions_list
    if event.kind == "pass":
        matches = [
            action
            for action in actions
            if isinstance(action, PassAction) and action.player_pos == event.player
        ]
    elif event.kind == "play":
        cards = frozenset(card_from_token(token) for token in event.cards)
        matches = [
            action
            for action in actions
            if isinstance(action, PlayCombination)
            and action.player_pos == event.player
            and frozenset(action.combination.cards) == cards
        ]
        if not matches:
            matches = _equivalent_physical_actions(state, event.player, cards)
    elif event.kind == "wish":
        rank = WISH_RANKS[event.value]
        matches = [
            action
            for action in actions
            if isinstance(action, WishAction)
            and action.player_pos == event.player
            and action.wish == rank
        ]
    elif event.kind == "dragon_to":
        matches = [
            action
            for action in actions
            if isinstance(action, GiveDragonAwayAction) and action.to == event.player
        ]
    else:
        return None, []
    return (matches[0] if matches else None), matches


def _rank_signature(cards):
    return tuple(sorted(card.rank.value for card in cards if card is not Card.PHOENIX))


def _equivalent_physical_actions(state, player, cards):
    """Recover legal suit variants omitted by the engine's action generator.

    For ordinary combinations suits do not determine strength, but the legacy
    generator emits one representative card per rank.  The actual physical
    cards still matter for future bombs, so replay constructs the exact logged
    combination after matching it to an otherwise equivalent legal action.
    """
    if not cards.issubset(state.handcards[player]):
        return []
    signature = _rank_signature(cards)
    recovered = []
    for candidate in state.possible_actions_list:
        if not isinstance(candidate, PlayCombination) or candidate.player_pos != player:
            continue
        if len(candidate.combination.cards) != len(cards):
            continue
        if _rank_signature(candidate.combination.cards) != signature:
            continue
        try:
            if isinstance(candidate.combination, Straight) and Card.PHOENIX in cards:
                combination = Straight(
                    cards, phoenix_as=candidate.combination.phoenix_as
                )
            elif isinstance(candidate.combination, FullHouse) and Card.PHOENIX in cards:
                combination = FullHouse.from_cards(
                    cards,
                    phoenix_as_trio_rank=candidate.combination.trio.rank,
                )
            else:
                combination = Combination.make(cards)
            if type(combination) is not type(candidate.combination):
                continue
            recovered.append(
                candidate.__class__(player_pos=player, combination=combination)
            )
        except (AssertionError, TypeError, ValueError):
            continue
    return recovered


def _apply_action(state, action):
    if action in state.possible_actions_set:
        return state.next_state(action)
    if isinstance(action, PlayCombination):
        # The action was recovered as a physical suit variant of a generated
        # legal combination.  The normal transition is still used; only the
        # legacy membership check is bypassed.
        return state._next_state_on_combination(action)
    raise ValueError("recovered non-play action is not legal")


def canonical_action(state, action):
    """Map an exact logged suit choice to the engine's legal representative."""
    if action in state.possible_actions_set:
        return action
    if not isinstance(action, PlayCombination):
        return None
    signature = _rank_signature(action.combination.cards)
    for candidate in state.possible_actions_list:
        if (
            isinstance(candidate, PlayCombination)
            and candidate.player_pos == action.player_pos
            and type(candidate.combination) is type(action.combination)
            and len(candidate.combination.cards) == len(action.combination.cards)
            and _rank_signature(candidate.combination.cards) == signature
            and candidate.combination.height == action.combination.height
        ):
            return candidate
    return None


def replay_round(round_, *, include_inferred_bomb_passes=False):
    """Replay one round and report the first incompatibility, if any."""
    if round_.scores is None:
        return ReplayResult(status="incomplete_round")
    issues = validate_game(type("Game", (), {"rounds": [round_]})())
    if issues:
        return ReplayResult(status="invalid_structure", detail=issues[0].code)

    try:
        state = make_initial_state(round_)
    except Exception as error:
        return ReplayResult(status="setup_error", detail=str(error))

    return _replay_from_event(
        round_,
        state,
        0,
        [],
        include_inferred_bomb_passes=include_inferred_bomb_passes,
    )


def _replay_from_event(
    round_, state, start_index, decisions, *, include_inferred_bomb_passes=False
):
    """Replay events, branching only when a logged Phoenix role is ambiguous."""

    ignored = {"grand_tichu", "tichu", "bomb_notice"}
    for event_index in range(start_index, len(round_.events)):
        event = round_.events[event_index]
        if event.kind in ignored:
            continue
        try:
            pending_kind = _action_kind(state.possible_actions_list)
            if (
                event.kind == "pass"
                and pending_kind in {"win_trick", "dragon"}
                and state.possible_actions_list[0].player_pos == event.player
            ):
                # BSW explicitly records the leading player's final pass.  In
                # this engine the trick has already finished at that point.
                continue
            state = _prepare_for_event(
                state,
                event,
                decisions if include_inferred_bomb_passes else None,
            )
            pending_kind = _action_kind(state.possible_actions_list)
            if (
                event.kind == "pass"
                and state.trick_on_table.is_empty()
                and state.player_pos == event.player
            ):
                # BSW may print the previous trick winner's final pass after
                # all hidden between-trick bomb declines were resolved.  A
                # pass cannot be a real action on an empty table.
                continue
            if (
                event.kind == "pass"
                and pending_kind in {"win_trick", "dragon"}
                and state.possible_actions_list[0].player_pos == event.player
            ):
                # A hidden bomb window may have been auto-declined above before
                # reaching BSW's explicit final leader pass.
                continue
            if state.is_terminal():
                if event.kind in {"wish", "dragon_to"}:
                    # BSW may print the administrative result of the last play
                    # after a double win has already ended the round.
                    continue
                return ReplayResult(
                    status="event_after_terminal",
                    decisions=decisions,
                    event_index=event_index,
                    detail=event.kind,
                )
            action, matches = _matching_action(state, event)
            if action is None:
                return ReplayResult(
                    status="illegal_{}".format(event.kind),
                    decisions=decisions,
                    event_index=event_index,
                    detail="player={} cards={} expected={}".format(
                        event.player,
                        event.cards,
                        _action_kind(state.possible_actions_list),
                    ),
                )
            if len(matches) > 1:
                branch_results = []
                for match in matches:
                    branch_decisions = list(decisions)
                    if event.kind in {"play", "pass"}:
                        branch_decisions.append((state, event, match))
                    try:
                        branch_state = _apply_action(state, match)
                    except Exception as error:
                        branch_results.append(
                            ReplayResult(
                                status="engine_error",
                                decisions=branch_decisions,
                                event_index=event_index,
                                detail="{}: {}".format(type(error).__name__, error),
                            )
                        )
                        continue
                    branch_results.append(
                        _replay_from_event(
                            round_,
                            branch_state,
                            event_index + 1,
                            branch_decisions,
                            include_inferred_bomb_passes=include_inferred_bomb_passes,
                        )
                    )

                return _best_branch_result(branch_results)

            if event.kind in {"play", "pass"}:
                decisions.append((state, event, action))
            state = _apply_action(state, action)
        except Exception as error:
            return ReplayResult(
                status="engine_error",
                decisions=decisions,
                event_index=event_index,
                detail="{}: {}".format(type(error).__name__, error),
            )

    try:
        state = _prepare_for_event(
            state,
            inferred_decisions=(
                decisions if include_inferred_bomb_passes else None
            ),
        )
        if not state.is_terminal():
            return ReplayResult(
                status="not_terminal",
                decisions=decisions,
                detail="ranking={} hand_sizes={}".format(
                    state.ranking, [len(hand) for hand in state.handcards]
                ),
            )
        points = state.count_points()
        actual = (points[0], points[1])
        status = "ok" if actual == round_.scores else "score_mismatch"
        return ReplayResult(
            status=status,
            decisions=decisions,
            expected_scores=round_.scores,
            actual_scores=actual,
            final_state=state,
        )
    except Exception as error:
        return ReplayResult(
            status="engine_error",
            decisions=decisions,
            detail="{}: {}".format(type(error).__name__, error),
        )


def _best_branch_result(results):
    """Prefer exact, then complete, then furthest-progressing replay branches."""
    exact = next((result for result in results if result.status == "ok"), None)
    if exact is not None:
        return exact
    completed = next(
        (result for result in results if result.status == "score_mismatch"), None
    )
    if completed is not None:
        return completed
    return max(
        results,
        key=lambda result: (
            result.event_index is not None,
            result.event_index if result.event_index is not None else -1,
        ),
    )


def _replay_path(path):
    """Replay one file and return only serialization-safe summary rows."""
    path = Path(path)
    game = parse_file(path)
    rows = []
    decisions = 0
    for round_index, round_ in enumerate(game.rounds):
        result = replay_round(round_)
        decisions += len(result.decisions)
        rows.append(
            {
                "game_id": int(path.stem),
                "round": round_index,
                "status": result.status,
                "event_index": result.event_index,
                "detail": result.detail,
                "decisions": len(result.decisions),
                "expected_scores": result.expected_scores,
                "actual_scores": result.actual_scores,
            }
        )
    return rows, decisions


def scan_directory(input_dir, report_path=None, *, show_progress=False, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    counts = Counter()
    games = rounds = decisions = 0
    output = None
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        output = report_path.open("w", encoding="utf-8", newline="\n")
    try:
        paths = sorted(Path(input_dir).glob("*.tch"))
        progress_step = max(1, len(paths) // 20)
        executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        results = (
            executor.map(_replay_path, paths, chunksize=1)
            if executor is not None
            else map(_replay_path, paths)
        )
        for path_index, (file_rows, file_decisions) in enumerate(results, start=1):
            games += 1
            decisions += file_decisions
            for row in file_rows:
                rounds += 1
                counts[row["status"]] += 1
                if output:
                    output.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
            if show_progress and (
                path_index == len(paths) or path_index % progress_step == 0
            ):
                percent = 100.0 * path_index / max(1, len(paths))
                print(
                    "[replay] {}/{} files ({:.0f}%)".format(
                        path_index, len(paths), percent
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if "executor" in locals() and executor is not None:
            executor.shutdown()
        if output:
            output.close()
    return {
        "games": games,
        "rounds": rounds,
        "decisions_before_failure": decisions,
        "counts": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Replay Brettspielwelt logs locally")
    parser.add_argument("--input-dir", default="datasets/raw/brettspielwelt")
    parser.add_argument(
        "--report", default="datasets/processed/brettspielwelt-replay-report-v2.jsonl"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    print(
        json.dumps(
            scan_directory(
                args.input_dir,
                args.report,
                show_progress=True,
                workers=args.workers,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
