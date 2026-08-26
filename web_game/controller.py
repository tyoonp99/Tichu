"""Server-authoritative controller for a two-human versus Model C round.

The browser only receives the current viewer's hand and opaque identifiers for
legal actions.  The authoritative ``TichuState`` never leaves this process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from time import monotonic
import uuid

import gymnasium as gym

import gym_tichu  # Registers the environment.
from gym_tichu.envs.internals import (
    Card,
    CardRank,
    CardTrade,
    DOG_COMBINATION,
    GiveDragonAwayAction,
    PassAction,
    PassBombAction,
    PlayCombination,
    TichuAction,
    WinTrickAction,
    WishAction,
    wishable_card_ranks,
)

HUMAN_SEATS = frozenset({0, 2})


class WebGameError(ValueError):
    """Raised when a browser requests an unavailable or illegal action."""


def automatic_trade(state: Any, player: int):
    """Choose deterministic, low-impact cards while the MVP trade UI is absent."""
    hand = sorted(state.handcards[player])
    return hand[0], hand[1], hand[-1]


def automatic_wish(state: Any, player: int) -> CardRank:
    """Choose a visible, deterministic wish while AI declaration policy is simple.

    The Model C policy only chooses card-play actions, so it has no learned
    Mahjong-wish head.  Until that exists, prefer the rank the AI holds most
    often (then the higher rank as a stable tie-breaker).  Crucially, this is a
    real Tichu wish rather than ``None`` so players can see and follow it.
    """
    ranks = tuple(sorted(wishable_card_ranks, key=lambda rank: rank.value))
    counts = {rank: 0 for rank in ranks}
    for card in state.handcards[player]:
        if card.rank in counts:
            counts[card.rank] += 1
    return max(ranks, key=lambda rank: (counts[rank], rank.value))


@dataclass
class WebGame:
    """One round shared by two human seats and two AI seats.

    Declarations and the three-card exchange are deliberately automatic in the
    prototype.  All play, wishes, and Dragon give-away decisions remain inside
    the rules engine and wait for a human when appropriate.
    """

    ai_factory: Callable[[], Any]
    seed: int | None = None
    human_seats: frozenset[int] = HUMAN_SEATS
    clock: Callable[[], float] = monotonic
    # Long enough to see a card reach the table, short enough that normal play
    # still feels like a click responds immediately.
    action_delay_seconds: float = 0.45
    bomb_window_seconds: float = 3.0
    env: Any = field(init=False)
    agents: dict[int, Any] = field(init=False)
    state: Any = field(init=False)
    terminal_points: tuple[int, int] | None = field(default=None, init=False)
    log: list[str] = field(default_factory=list, init=False)
    action_ids: dict[str, Any] = field(default_factory=dict, init=False)
    next_step_at: float = field(default=0.0, init=False)
    bomb_deadline: float | None = field(default=None, init=False)
    last_event: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.human_seats or not self.human_seats.issubset({0, 1, 2, 3}):
            raise ValueError("human_seats must be a non-empty subset of 0..3")
        self.env = gym.make("tichu_multiplayer-v0", verbose=False)
        self.agents = {
            player: self.ai_factory()
            for player in range(4)
            if player not in self.human_seats
        }
        self._setup_round()
        self.progress()

    def _setup_round(self):
        state, _ = self.env.reset(seed=self.seed)
        # MVP policy: no Grand/normal Tichu calls.  Human declarations can be
        # added later without changing the server/client trust boundary.
        state, _, terminated, truncated, _ = self.env.step(set())
        assert not terminated and not truncated
        state, _, terminated, truncated, _ = self.env.step(set())
        assert not terminated and not truncated
        trades = []
        for player in range(4):
            left, partner, right = automatic_trade(state, player)
            trades.extend(
                (
                    CardTrade(player, (player - 1) % 4, left),
                    CardTrade(player, (player + 2) % 4, partner),
                    CardTrade(player, (player + 1) % 4, right),
                )
            )
        self.state, _, terminated, truncated, _ = self.env.step(trades)
        assert not terminated and not truncated
        self.log.append("라운드를 시작했습니다. 티츄 선언과 카드 교환은 이 시험판에서 자동 처리됩니다.")

    def _apply(
        self,
        action: Any,
        *,
        delay: float | None = None,
        keep_bomb_deadline: bool = False,
        record_event: bool = True,
    ):
        actor = action.player_pos
        self.state, reward, terminated, truncated, _ = self.env.step(action)
        if record_event and isinstance(action, PlayCombination):
            self.log.append("플레이어 {}: {}".format(actor, action.combination))
        elif record_event and isinstance(action, (PassAction, PassBombAction)):
            self.log.append("플레이어 {}: 패스".format(actor))
        if terminated or truncated:
            self.terminal_points = (reward[0], reward[1])
            self.log.append("라운드 종료: 인간 팀 {}점 / AI 팀 {}점".format(*self.terminal_points))
        if record_event:
            self.last_event = {
                "actor": actor,
                "kind": action.__class__.__name__,
                "label": self._action_label(action),
                "cards": self._action_cards(action),
            }
        self.next_step_at = self.clock() + (self.action_delay_seconds if delay is None else delay)
        if not keep_bomb_deadline:
            self.bomb_deadline = None
            # Every ordinary play gets one shared reaction window.  The UI does
            # not identify it as a bomb window, but a player who actually has a
            # legal bomb can use that time to play it.  Mahjong is the one
            # exception: its wish must be chosen before reactions can begin.
            if (
                isinstance(action, PlayCombination)
                and action.combination != DOG_COMBINATION
                and Card.MAHJONG not in action.combination
            ):
                self._start_reaction_window()

    def _start_reaction_window(self):
        self.bomb_deadline = self.clock() + self.bomb_window_seconds

    def _first_action(self):
        return self.state.possible_actions_list[0]

    def _human_decision_player(self) -> int | None:
        if self.terminal_points is not None:
            return None
        first = self._first_action()
        if isinstance(first, (WishAction, GiveDragonAwayAction)):
            return first.player_pos if first.player_pos in self.human_seats else None
        if isinstance(first, (PassAction, PassBombAction, PlayCombination)):
            return first.player_pos if first.player_pos in self.human_seats else None
        return None

    def waiting_player(self) -> int | None:
        """Human that may act now; returns None during an on-table animation."""
        if self.clock() < self.next_step_at:
            return None
        if self.bomb_deadline is not None and self.clock() >= self.bomb_deadline:
            return None
        return self._human_decision_player()

    def progress(self):
        """Advance at most one forced/AI action when its display delay has elapsed."""
        if self.terminal_points is not None or self.clock() < self.next_step_at:
            return

        first = self._first_action()
        human = self._human_decision_player()

        # A play with no legal bomb response still keeps the same three-second
        # pause.  That makes the rhythm predictable and gives the browser a
        # chance to expose a bomb when the engine does have one.
        if (
            self.bomb_deadline is not None
            and not isinstance(first, (PassBombAction, WishAction))
        ):
            if self.clock() < self.bomb_deadline:
                return
            self.bomb_deadline = None
            return

        if isinstance(first, PassBombAction):
            if self.bomb_deadline is None:
                self.bomb_deadline = self.clock() + self.bomb_window_seconds
                return

            # One shared reaction window covers everyone with a legal bomb.
            # The rules engine represents those players sequentially, but the
            # web controller silently walks its internal "no" responses rather
            # than giving every seat a fresh three-second countdown.
            if self.clock() >= self.bomb_deadline:
                while (
                    self.terminal_points is None
                    and isinstance(self._first_action(), PassBombAction)
                ):
                    self._apply(
                        self._first_action(),
                        delay=0,
                        keep_bomb_deadline=True,
                        record_event=False,
                    )
                self.bomb_deadline = None
                return
            if human is not None:
                return

            action = self.agents[first.player_pos].action(self.state)
            if isinstance(action, PassBombAction):
                self._apply(
                    action,
                    delay=0,
                    keep_bomb_deadline=True,
                    record_event=False,
                )
            else:
                self._apply(action)
            return
        if human is not None:
            return

        if isinstance(first, TichuAction):
            self._apply(TichuAction(first.player_pos, announce_tichu=False))
        elif isinstance(first, WishAction):
            self._apply(WishAction(first.player_pos, wish=automatic_wish(self.state, first.player_pos)))
            self._start_reaction_window()
        elif isinstance(first, GiveDragonAwayAction):
            receiver = self.agents[first.player_pos].give_dragon_away(
                self.state, first.player_pos
            )
            self._apply(GiveDragonAwayAction(first.player_pos, receiver, first.trick))
        elif isinstance(first, WinTrickAction):
            self._apply(first)
        elif isinstance(first, (PassAction, PassBombAction, PlayCombination)):
            self._apply(self.agents[first.player_pos].action(self.state))
        else:  # pragma: no cover - protects a future engine action type.
            raise RuntimeError("Unsupported engine action: {!r}".format(first))

    def _legal_actions_for(self, player: int) -> list[Any]:
        if self.waiting_player() != player:
            return []
        first = self._first_action()
        if isinstance(first, (WishAction, GiveDragonAwayAction)):
            return []
        return list(self.state.possible_actions_list)

    @staticmethod
    def _action_label(action: Any) -> str:
        if isinstance(action, (PassAction, PassBombAction)):
            return "패스"
        if isinstance(action, PlayCombination):
            return str(action.combination)
        return str(action)

    @staticmethod
    def _action_cards(action: Any) -> list[str]:
        if isinstance(action, PlayCombination):
            return [str(card) for card in sorted(action.combination.cards)]
        return []

    def view_for(self, player: int) -> dict[str, Any]:
        if player not in self.human_seats:
            raise WebGameError("이 방에서 선택한 사람 좌석이 아닙니다.")
        self.progress()
        legal = []
        for action in self._legal_actions_for(player):
            action_id = uuid.uuid4().hex
            self.action_ids[action_id] = action
            legal.append(
                {
                    "id": action_id,
                    "label": self._action_label(action),
                    "cards": self._action_cards(action),
                    "kind": (
                        action.combination.__class__.__name__
                        if isinstance(action, PlayCombination)
                        else action.__class__.__name__
                    ),
                }
            )

        first = self._first_action() if self.terminal_points is None else None
        table_action = (
            self.state.trick_on_table.last_combination_action
            if not self.state.trick_on_table.is_empty()
            else None
        )
        prompt = None
        if self.waiting_player() == player and isinstance(first, WishAction):
            prompt = {"kind": "wish", "choices": [rank.name for rank in wishable_card_ranks]}
        elif self.waiting_player() == player and isinstance(first, GiveDragonAwayAction):
            prompt = {
                "kind": "dragon",
                "choices": [(player - 1) % 4, (player + 1) % 4],
                "points": first.trick.points,
            }

        return {
            "seat": player,
            "partner": (player + 2) % 4,
            "waiting_player": self.waiting_player(),
            "current_player": self.state.player_pos,
            "turn_kind": first.__class__.__name__ if first is not None else None,
            "animation_remaining": max(0.0, self.next_step_at - self.clock()),
            # This is a shared pause after every ordinary card play, not a
            # signal that any particular player owns a bomb.
            "reaction_remaining": (
                max(0.0, self.bomb_deadline - self.clock())
                if self.bomb_deadline is not None
                else None
            ),
            "last_event": self.last_event,
            "hand": [str(card) for card in sorted(self.state.handcards[player])],
            "hand_sizes": [len(cards) for cards in self.state.handcards],
            "ranking": list(self.state.ranking),
            "table": str(self.state.trick_on_table.last_combination) if not self.state.trick_on_table.is_empty() else None,
            "table_cards": (
                [str(card) for card in sorted(self.state.trick_on_table.last_combination.cards)]
                if not self.state.trick_on_table.is_empty()
                else []
            ),
            "table_player": table_action.player_pos if table_action is not None else None,
            "table_points": self.state.trick_on_table.points if table_action is not None else 0,
            "wish": str(self.state.wish) if self.state.wish else None,
            "legal_actions": legal,
            "prompt": prompt,
            "terminal_points": self.terminal_points,
            "log": self.log[-12:],
        }

    def play(self, player: int, action_id: str):
        self.progress()
        if self.waiting_player() != player:
            raise WebGameError("지금은 당신의 차례가 아닙니다.")
        action = self.action_ids.get(action_id)
        if action is None or action not in self._legal_actions_for(player):
            raise WebGameError("유효하지 않거나 만료된 행동입니다. 화면을 새로고침하세요.")
        if isinstance(action, PassBombAction):
            self._apply(
                action,
                delay=0,
                keep_bomb_deadline=True,
                record_event=False,
            )
        else:
            self._apply(action)

    def choose_prompt(self, player: int, choice: str | int):
        self.progress()
        if self.waiting_player() != player:
            raise WebGameError("지금은 당신의 결정 차례가 아닙니다.")
        first = self._first_action()
        if isinstance(first, WishAction):
            try:
                rank = CardRank.from_name(str(choice))
            except ValueError as error:
                raise WebGameError("유효하지 않은 마작 소원입니다.") from error
            self._apply(WishAction(player, rank))
            self._start_reaction_window()
        elif isinstance(first, GiveDragonAwayAction):
            receiver = int(choice)
            if receiver not in {(player - 1) % 4, (player + 1) % 4}:
                raise WebGameError("드래곤 트릭은 상대에게만 줄 수 있습니다.")
            self._apply(GiveDragonAwayAction(player, receiver, first.trick))
        else:
            raise WebGameError("현재 선택할 특별 행동이 없습니다.")
