"""Fixed-size features for supervised Tichu action ranking.

This module deliberately has no neural-network dependency.  It converts the
privacy-safe JSONL records into NumPy vectors and is shared by training and
inference code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


def _sorted_values(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(value for value in values if value is not None))


@dataclass(frozen=True)
class FeatureSchema:
    """Categorical vocabularies and fixed-window feature dimensions."""

    cards: tuple[str, ...]
    action_types: tuple[str, ...]
    combinations: tuple[str, ...]
    wishes: tuple[str, ...]
    max_trick_actions: int = 12

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping], *, max_trick_actions: int = 12
    ) -> "FeatureSchema":
        cards: set[str] = set()
        action_types: set[str] = set()
        combinations: set[str] = set()
        wishes: set[str] = set()

        def observe_action(action: Mapping) -> None:
            action_types.add(action.get("type", "UNKNOWN"))
            combinations.add(action.get("combination", "NONE"))
            cards.update(action.get("cards", ()))
            wish = action.get("wish")
            if wish is not None:
                wishes.add(wish)

        for record in records:
            observation = record["observation"]
            cards.update(observation.get("hand", ()))
            wish = observation.get("wish")
            if wish is not None:
                wishes.add(wish)
            for action in observation.get("trick", ()):
                observe_action(action)
            for action in record["legal_actions"]:
                observe_action(action)
            observe_action(record["chosen_action"])

        return cls(
            cards=_sorted_values(cards),
            action_types=_sorted_values(action_types),
            combinations=_sorted_values(combinations),
            wishes=_sorted_values(wishes),
            max_trick_actions=max_trick_actions,
        )

    @classmethod
    def from_jsonl(
        cls, path: str | Path, *, max_trick_actions: int = 12
    ) -> "FeatureSchema":
        return cls.from_records(iter_jsonl(path), max_trick_actions=max_trick_actions)

    @classmethod
    def from_dict(cls, value: Mapping) -> "FeatureSchema":
        return cls(
            cards=tuple(value["cards"]),
            action_types=tuple(value["action_types"]),
            combinations=tuple(value["combinations"]),
            wishes=tuple(value["wishes"]),
            max_trick_actions=int(value["max_trick_actions"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


class FeatureEncoder:
    """Encode observations and variable legal-action candidates."""

    def __init__(self, schema: FeatureSchema):
        self.schema = schema
        self._card = {value: index for index, value in enumerate(schema.cards)}
        self._action_type = {
            value: index for index, value in enumerate(schema.action_types)
        }
        self._combination = {
            value: index for index, value in enumerate(schema.combinations)
        }
        self._wish = {value: index for index, value in enumerate(schema.wishes)}

        self.action_size = (
            len(schema.action_types)
            + 4
            + len(schema.combinations)
            + len(schema.cards)
            + 1  # height
            + 2  # announce, grand
            + len(schema.wishes)
            + 1  # no-wish marker
            + 4  # dragon recipient
            + 1  # trick points
        )
        public_size = (
            len(schema.cards)
            + 4  # hand sizes
            + 4  # won trick counts
            + 4  # won trick points
            + len(schema.wishes)
            + 1  # no-wish marker
            + 4  # ranking
            + 4  # announced tichu
            + 4  # announced grand tichu
        )
        self.state_size = public_size + schema.max_trick_actions * self.action_size

    @staticmethod
    def _one_hot(vector: np.ndarray, offset: int, index: int | None) -> None:
        if index is not None:
            vector[offset + index] = 1.0

    def encode_action(self, action: Mapping) -> np.ndarray:
        vector = np.zeros(self.action_size, dtype=np.float32)
        offset = 0

        self._one_hot(vector, offset, self._action_type.get(action.get("type")))
        offset += len(self.schema.action_types)

        player = action.get("player")
        self._one_hot(vector, offset, player if player in range(4) else None)
        offset += 4

        combination = action.get("combination", "NONE")
        self._one_hot(vector, offset, self._combination.get(combination))
        offset += len(self.schema.combinations)

        for card in action.get("cards", ()):
            self._one_hot(vector, offset, self._card.get(card))
        offset += len(self.schema.cards)

        height = action.get("height")
        vector[offset] = 0.0 if height is None else float(height) / 15.0
        offset += 1

        vector[offset] = float(bool(action.get("announce", False)))
        vector[offset + 1] = float(bool(action.get("grand", False)))
        offset += 2

        wish = action.get("wish")
        self._one_hot(vector, offset, self._wish.get(wish))
        offset += len(self.schema.wishes)
        vector[offset] = float(wish is None)
        offset += 1

        recipient = action.get("to")
        self._one_hot(vector, offset, recipient if recipient in range(4) else None)
        offset += 4

        vector[offset] = float(action.get("trick_points", 0)) / 100.0
        return vector

    def encode_observation(self, observation: Mapping) -> np.ndarray:
        vector = np.zeros(self.state_size, dtype=np.float32)
        offset = 0

        for card in observation.get("hand", ()):
            self._one_hot(vector, offset, self._card.get(card))
        offset += len(self.schema.cards)

        for values, scale in (
            (observation.get("hand_sizes", ()), 14.0),
            (observation.get("won_trick_counts", ()), 14.0),
            (observation.get("won_trick_points", ()), 200.0),
        ):
            for index, value in enumerate(values[:4]):
                vector[offset + index] = float(value) / scale
            offset += 4

        wish = observation.get("wish")
        self._one_hot(vector, offset, self._wish.get(wish))
        offset += len(self.schema.wishes)
        vector[offset] = float(wish is None)
        offset += 1

        for key in ("ranking", "announced_tichu", "announced_grand_tichu"):
            for position in observation.get(key, ()):
                self._one_hot(
                    vector, offset, position if position in range(4) else None
                )
            offset += 4

        trick = observation.get("trick", ())[-self.schema.max_trick_actions :]
        # Right-align so the newest public actions always occupy the same slots.
        trick_offset = self.state_size - len(trick) * self.action_size
        for index, action in enumerate(trick):
            start = trick_offset + index * self.action_size
            vector[start : start + self.action_size] = self.encode_action(action)
        return vector

    def encode_record(self, record: Mapping) -> tuple[np.ndarray, np.ndarray, int]:
        legal_actions = record["legal_actions"]
        if not legal_actions:
            raise ValueError("record has no legal actions")
        try:
            chosen_index = legal_actions.index(record["chosen_action"])
        except ValueError as error:
            raise ValueError("chosen action is absent from legal actions") from error
        candidates = np.stack([self.encode_action(action) for action in legal_actions])
        return self.encode_observation(record["observation"]), candidates, chosen_index


def buffered_shuffle(
    records: Iterable[dict], *, buffer_size: int, rng: np.random.Generator
) -> Iterator[dict]:
    """Bounded-memory shuffle suitable for large JSONL files."""
    if buffer_size <= 1:
        yield from records
        return
    buffer: list[dict] = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = int(rng.integers(len(buffer)))
        yield buffer[index]
        buffer[index] = record
    rng.shuffle(buffer)
    yield from buffer


def batches(records: Iterable[dict], batch_size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
