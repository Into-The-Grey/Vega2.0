import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset, DataLoader

import importlib.util
import pathlib

_DU_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "vega" / "federated" / "data_utils.py"
import sys
_DU_SPEC = importlib.util.spec_from_file_location("vega_data_utils", _DU_PATH)
data_utils = importlib.util.module_from_spec(_DU_SPEC)
sys.modules["vega_data_utils"] = data_utils
assert _DU_SPEC and _DU_SPEC.loader
_DU_SPEC.loader.exec_module(data_utils)  # type: ignore[attr-defined]

partition_dataset = data_utils.partition_dataset
compute_data_statistics = data_utils.compute_data_statistics


def _build_tiny_dataset(num_samples: int = 20):
    features = torch.arange(num_samples, dtype=torch.float32).unsqueeze(1)
    labels = torch.cat(
        [torch.zeros(num_samples // 2, dtype=torch.long), torch.ones(num_samples // 2, dtype=torch.long)]
    )
    return TensorDataset(features, labels)


def test_partition_dataset_iid_balanced():
    dataset = _build_tiny_dataset()
    partitions = partition_dataset(dataset, num_partitions=4, strategy="iid", seed=7)

    assert len(partitions) == 4
    total_indices = sum(len(part.indices) for part in partitions)
    assert total_indices == len(dataset)

    sizes = [len(part.indices) for part in partitions]
    assert max(sizes) - min(sizes) <= 1


def test_partition_dataset_non_iid_label_skew():
    dataset = _build_tiny_dataset()
    partitions = partition_dataset(
        dataset,
        num_partitions=2,
        strategy="non_iid",
        seed=3,
        shards_per_participant=1,
    )

    assert len(partitions) == 2
    label_sets = [set(part.label_distribution.keys()) for part in partitions]
    # Each participant should receive primarily one label shard
    assert all(len(labels) == 1 for labels in label_sets)
    assert label_sets[0] != label_sets[1]


def test_compute_data_statistics_matches_loader():
    dataset = _build_tiny_dataset()
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    stats = compute_data_statistics(loader)

    assert stats.sample_count == len(dataset)
    assert pytest.approx(stats.feature_mean, rel=1e-3) == dataset[:][0].mean().item()
    assert stats.label_distribution["0"] == len(dataset) // 2
    assert stats.label_distribution["1"] == len(dataset) // 2
