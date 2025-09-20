import numpy as np

from src.vega.federated.fedavg import FedAvg, FedAvgConfig, AsyncAggregator


def test_async_aggregator_simple():
    cfg = FedAvgConfig()
    fa = FedAvg(cfg)
    t = [0.0]

    def now():
        return t[0]

    agg = AsyncAggregator(min_updates=2, timeout_seconds=5.0, time_provider=now)
    agg.start_round("r1")
    agg.submit_update({"w": np.array([1.0, 1.0])}, 10)
    assert not agg.is_ready()
    agg.submit_update({"w": np.array([3.0, 3.0])}, 30)
    assert agg.is_ready()
    out = agg.aggregate(fa)
    assert np.allclose(out[0]["w"], np.array([2.5, 2.5]))


def test_byr_robust_median():
    cfg = FedAvgConfig(byzantine_tolerance=True, byzantine_method="median")
    fa = FedAvg(cfg)
    updates = [
        {"w": np.array([1.0, 1.0])},
        {"w": np.array([2.0, 2.0])},
        {"w": np.array([100.0, 100.0])},
    ]
    agg = fa.byzantine_robust_aggregate(updates, method="median")
    assert np.allclose(agg["w"], np.array([2.0, 2.0]))


if __name__ == "__main__":
    test_async_aggregator_simple()
    test_byr_robust_median()
    print("FedAvg/Async/Byzantine tests passed")
