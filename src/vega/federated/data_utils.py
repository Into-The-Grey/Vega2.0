"""Utility helpers for preparing federated learning datasets.

Provides reusable functions for:

* Partitioning datasets into IID or non-IID shards for participants
* Computing dataset statistics without sharing raw data
* Lightweight metadata containers that the coordinator/participants can
  exchange for compatibility and monitoring

These utilities are intentionally framework-agnostic. When PyTorch is
available we return ``torch.utils.data.Subset`` instances for convenient
training integration; otherwise we fall back to simple index lists to keep
the helpers usable in pure NumPy or TensorFlow pipelines.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np

try:  # Optional PyTorch support
    import torch
    from torch.utils.data import Dataset, Subset, DataLoader

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch optional for runtime
    torch = None  # type: ignore
    Dataset = Any  # type: ignore
    Subset = Any  # type: ignore
    DataLoader = Any  # type: ignore
    HAS_TORCH = False


@dataclass
class DataStatistics:
    """Summary statistics for a dataset or data loader."""

    sample_count: int
    feature_mean: Optional[float]
    feature_std: Optional[float]
    feature_min: Optional[float]
    feature_max: Optional[float]
    label_distribution: dict[str, int]
    feature_dtype: Optional[str] = None


@dataclass
class PartitionResult:
    """Represents a partitioned dataset for a participant."""

    indices: List[int]
    subset: Any
    label_distribution: dict[str, int]


def _default_label_getter(dataset: Any) -> Callable[[int], Any]:
    """Infer a label getter for common dataset types."""

    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")

        def _getter(idx: int) -> Any:
            target = targets[idx]
            if HAS_TORCH and torch is not None and isinstance(target, torch.Tensor):
                return target.item() if target.numel() == 1 else tuple(target.tolist())
            if isinstance(target, np.ndarray):
                return target.item() if target.size == 1 else tuple(target.tolist())
            return target

        return _getter

    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")

        def _getter(idx: int) -> Any:
            return labels[idx]

        return _getter

    # Handle common PyTorch pattern: TensorDataset(features, labels)
    if HAS_TORCH and torch is not None and hasattr(dataset, "tensors"):
        tensors = getattr(dataset, "tensors")
        if isinstance(tensors, (list, tuple)) and len(tensors) >= 2:
            label_tensor = tensors[1]

            def _getter(idx: int) -> Any:
                val = label_tensor[idx]
                if torch.is_tensor(val):
                    return val.item() if val.numel() == 1 else tuple(val.tolist())
                if isinstance(val, np.ndarray):
                    return val.item() if val.size == 1 else tuple(val.tolist())
                return val

            return _getter

    raise ValueError(
        "Unable to infer label getter. Provide label_getter for non-standard dataset."
    )


def partition_dataset(
    dataset: Dataset | Sequence[Any],
    num_partitions: int,
    *,
    strategy: str = "iid",
    seed: Optional[int] = None,
    label_getter: Optional[Callable[[int], Any]] = None,
    shards_per_participant: int = 2,
) -> List[PartitionResult]:
    """Split a dataset into multiple partitions.

    Args:
        dataset: Any indexable dataset (PyTorch Dataset, list, numpy array, ...).
        num_partitions: Number of partitions/participants required.
        strategy: ``"iid"`` for random balanced shards, ``"non_iid"`` for
            label-skewed shards that simulate heterogeneous participants.
        seed: Optional random seed for reproducibility.
        label_getter: Callable returning the label for a given index. Required
            for ``strategy="non_iid"`` when labels cannot be inferred.
        shards_per_participant: Number of label shards to assign to each
            participant when using ``"non_iid"`` strategy.

    Returns:
        List of :class:`PartitionResult` instances.
    """

    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")

    total_samples = len(dataset)  # type: ignore[arg-type]
    if total_samples == 0:
        raise ValueError("Cannot partition an empty dataset")

    rng = random.Random(seed)
    indices = list(range(total_samples))

    if strategy.lower() in {"iid", "uniform"}:
        rng.shuffle(indices)
        splits = np.array_split(indices, num_partitions)
        results: List[PartitionResult] = []
        for split in splits:
            split_indices = split.astype(int).tolist()
            label_distribution = _compute_label_distribution(
                dataset, split_indices, label_getter
            )
            subset = _create_subset(dataset, split_indices)
            results.append(
                PartitionResult(
                    indices=split_indices,
                    subset=subset,
                    label_distribution=label_distribution,
                )
            )
        return results

    if strategy.lower() not in {"non_iid", "label_shard"}:
        raise ValueError(f"Unknown partition strategy: {strategy}")

    # Non-IID partitioning: group by labels and assign shards.
    getter = label_getter or _default_label_getter(dataset)
    label_buckets: dict[Any, List[int]] = defaultdict(list)
    for idx in indices:
        label_buckets[getter(idx)].append(idx)

    shards: List[List[int]] = []
    for label, bucket in label_buckets.items():
        rng.shuffle(bucket)
        shard_size = max(1, math.ceil(len(bucket) / shards_per_participant))
        for start in range(0, len(bucket), shard_size):
            shards.append(bucket[start : start + shard_size])

    if not shards:
        raise ValueError("No shards produced; check label distribution and parameters")

    rng.shuffle(shards)
    required_shards = num_partitions * shards_per_participant

    # If there are not enough shards, recycle existing ones to meet demand.
    if len(shards) < required_shards:
        multiplier = math.ceil(required_shards / len(shards))
        shards = (shards * multiplier)[:required_shards]

    partitions = [[] for _ in range(num_partitions)]
    for shard_index, shard in enumerate(shards[:required_shards]):
        participant_idx = shard_index % num_partitions
        partitions[participant_idx].extend(shard)

    results = []
    for participant_indices in partitions:
        rng.shuffle(participant_indices)
        label_distribution = _compute_label_distribution(
            dataset, participant_indices, label_getter
        )
        subset = _create_subset(dataset, participant_indices)
        results.append(
            PartitionResult(
                indices=list(participant_indices),
                subset=subset,
                label_distribution=label_distribution,
            )
        )

    return results


def compute_data_statistics(
    data_source: Iterable,
    *,
    max_batches: Optional[int] = None,
    label_transform: Optional[Callable[[Any], Any]] = None,
) -> DataStatistics:
    """Compute summary statistics from a data loader or iterable.

    Only aggregated values are returned—no raw samples are retained—so the
    results can be safely shared across participants and the coordinator.

    Args:
        data_source: Iterable yielding ``(features, labels)`` or ``features``.
        max_batches: Optional cap on the number of batches inspected. Useful for
            very large datasets when only approximate statistics are required.
        label_transform: Optional callable to normalise label values (e.g.,
            ``int`` casting).

    Returns:
        :class:`DataStatistics` containing aggregated metrics.
    """

    batch_counter = 0
    feature_sum = 0.0
    feature_sq_sum = 0.0
    feature_min = math.inf
    feature_max = -math.inf
    feature_dtype: Optional[str] = None
    value_count = 0
    label_counts: Counter[str] = Counter()

    iterator: Iterator[Any] = iter(data_source)

    for batch in iterator:
        if max_batches is not None and batch_counter >= max_batches:
            break

        batch_counter += 1

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            features, labels = batch[0], batch[1]
        else:
            features, labels = batch, None

        feature_array = _to_numpy(features)
        if feature_array is not None:
            flattened = feature_array.astype(np.float64, copy=False).reshape(-1)
            feature_sum += float(flattened.sum())
            feature_sq_sum += float(np.square(flattened).sum())
            value_count += flattened.size
            feature_min = float(min(feature_min, float(flattened.min())))
            feature_max = float(max(feature_max, float(flattened.max())))
            if feature_dtype is None:
                feature_dtype = str(feature_array.dtype)

        if labels is not None:
            label_array = _to_numpy(labels)
            if label_array is None:
                label_items = _ensure_iterable(labels)
            else:
                label_items = label_array.reshape(-1)

            for item in label_items:
                value = item
                if label_transform is not None:
                    value = label_transform(item)
                elif HAS_TORCH and isinstance(item, np.generic):
                    value = item.item()
                elif HAS_TORCH and torch is not None and torch.is_tensor(item):  # type: ignore[union-attr]
                    value = item.item()

                label_counts[str(value)] += 1

    sample_count = _infer_sample_count(label_counts, value_count)

    if value_count == 0:
        feature_mean = feature_std = feature_min_val = feature_max_val = None
    else:
        feature_mean = feature_sum / value_count
        variance = max(feature_sq_sum / value_count - feature_mean**2, 0.0)
        feature_std = math.sqrt(variance)
        feature_min_val = feature_min if feature_min != math.inf else None
        feature_max_val = feature_max if feature_max != -math.inf else None

    return DataStatistics(
        sample_count=sample_count,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_min=feature_min_val,
        feature_max=feature_max_val,
        label_distribution=dict(label_counts),
        feature_dtype=feature_dtype,
    )


def _compute_label_distribution(
    dataset: Any, indices: Sequence[int], label_getter: Optional[Callable[[int], Any]]
) -> dict[str, int]:
    if not indices:
        return {}

    getter = label_getter or _default_label_getter(dataset)
    counter: Counter[str] = Counter()
    for idx in indices:
        label = getter(idx)
        counter[str(label)] += 1
    return dict(counter)


def _create_subset(dataset: Any, indices: Sequence[int]) -> Any:
    if not indices:
        return []

    if HAS_TORCH and torch is not None and isinstance(dataset, Dataset):  # type: ignore[arg-type]
        return Subset(dataset, list(indices))

    # Generic fallback – return the indices themselves for non-PyTorch datasets.
    return list(indices)


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if HAS_TORCH and torch is not None and torch.is_tensor(value):  # type: ignore[union-attr]
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
        return np.asarray(value)
    return None


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    if HAS_TORCH and torch is not None and torch.is_tensor(value):  # type: ignore[union-attr]
        return value.detach().cpu().numpy().reshape(-1)
    return [value]


def _infer_sample_count(label_counts: Counter[str], feature_value_count: int) -> int:
    if label_counts:
        return sum(label_counts.values())
    if feature_value_count:
        # We cannot infer batch size reliably; assume features represent scalar per sample.
        return feature_value_count
    return 0


__all__ = [
    "DataStatistics",
    "PartitionResult",
    "partition_dataset",
    "compute_data_statistics",
]
