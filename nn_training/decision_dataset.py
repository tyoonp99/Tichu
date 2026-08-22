"""Streaming PyTorch data loading for compressed Tichu decision JSONL."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from nn_training.behavior_cloning import PaddedCandidateBatch, encode_padded_batch
from nn_training.imitation_features import buffered_shuffle, iter_jsonl
from nn_training.imitation_features import FeatureEncoder


class JsonlDecisionDataset(IterableDataset):
    """Yield one decision record at a time without loading a split into memory.

    A JSONL record has a variable number of legal actions, so batching is
    deliberately deferred to the collate_decision_records function. Every
    worker opens its own gzip stream and receives a disjoint record subset.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_records: int | None = None,
        shuffle_buffer: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        if max_records is not None and max_records < 1:
            raise ValueError("max_records must be positive when provided")
        if shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be positive")
        self.path = Path(path)
        self.max_records = max_records
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _records_for_worker(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1

        for record_index, record in enumerate(iter_jsonl(self.path)):
            if self.max_records is not None and record_index >= self.max_records:
                return
            if record_index % worker_count == worker_id:
                yield record

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        records = self._records_for_worker()
        if self.shuffle_buffer > 1:
            records = buffered_shuffle(
                records,
                buffer_size=self.shuffle_buffer,
                rng=np.random.default_rng(self.seed + worker_id),
            )
        yield from records


def collate_decision_records(records: list[dict]) -> list[dict]:
    """Keep variable-length legal-action lists intact for later tensor encoding."""
    if not records:
        raise ValueError("cannot collate an empty decision batch")
    return list(records)


def make_decision_dataloader(
    path: str | Path,
    *,
    batch_size: int,
    max_records: int | None = None,
    shuffle_buffer: int = 1,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Create a deterministic streaming loader for one train or validation split."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    dataset = JsonlDecisionDataset(
        path,
        max_records=max_records,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_decision_records,
    )


def collate_encoded_decision_records(
    records: list[dict], *, encoder: FeatureEncoder
) -> PaddedCandidateBatch:
    """Encode a variable-length raw batch into tensors for Baseline B."""
    return encode_padded_batch(records, encoder)


def make_tensor_dataloader(
    path: str | Path,
    *,
    encoder: FeatureEncoder,
    batch_size: int,
    max_records: int | None = None,
    shuffle_buffer: int = 1,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Create a streaming DataLoader that yields PaddedCandidateBatch tensors."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    dataset = JsonlDecisionDataset(
        path,
        max_records=max_records,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate_encoded_decision_records, encoder=encoder),
    )
