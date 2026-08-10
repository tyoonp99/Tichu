import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

from gym_tichu.envs.internals.cards import GeneralCombination


PathLike = Union[str, Path]


class CombinationWeights:
    """Stores observed general-combination frequencies by hand size."""

    def __init__(self, filename: Optional[PathLike]=None, load: bool=True):
        self.filename = Path(filename) if filename is not None else Path("gcombweights.pkl")
        self._gcomb_counter: Dict[Tuple[int, GeneralCombination], int] = defaultdict(int)
        self._total_counter: Dict[int, int] = defaultdict(int)
        if load:
            self.load_from_file()

    @property
    def weights(self):
        result = defaultdict(float)
        for length_and_combination, count in self._gcomb_counter.items():
            hand_length = length_and_combination[0]
            total = self._total_counter[hand_length]
            if total:
                result[length_and_combination] = count / total
        return result

    @classmethod
    def weights_from_file(cls, filename: PathLike):
        weights = cls(filename=filename, load=False)
        weights.load_from_file()
        return weights.weights

    def add_combinations(self, hand_length: int, combinations: Iterable[GeneralCombination]):
        combinations = tuple(combinations)
        self._total_counter[hand_length] += len(combinations)
        for combination in combinations:
            self._gcomb_counter[(hand_length, combination)] += 1

    def probability(self, hand_length: int, combination: GeneralCombination)->float:
        total = self._total_counter[hand_length]
        if not total:
            return 0.0
        return self._gcomb_counter[(hand_length, combination)] / total

    def load_from_file(self, filename: Optional[PathLike]=None):
        source = Path(filename) if filename is not None else self.filename
        with source.open("rb") as file:
            self._gcomb_counter, self._total_counter = pickle.load(file)

    def save_to_file(self, filename: Optional[PathLike]=None):
        destination = Path(filename) if filename is not None else self.filename
        with destination.open("wb") as file:
            pickle.dump([self._gcomb_counter, self._total_counter], file)
