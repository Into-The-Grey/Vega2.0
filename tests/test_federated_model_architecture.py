import pytest

torch = pytest.importorskip("torch")
import torch
import torch.nn as nn

import importlib.util
import pathlib

_MS_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "vega" / "federated" / "model_serialization.py"
import sys
_MS_SPEC = importlib.util.spec_from_file_location("vega_model_serialization", _MS_PATH)
ms = importlib.util.module_from_spec(_MS_SPEC)
sys.modules["vega_model_serialization"] = ms
assert _MS_SPEC and _MS_SPEC.loader
_MS_SPEC.loader.exec_module(ms)  # type: ignore[attr-defined]

ModelSerializer = ms.ModelSerializer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class AlternateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_model_architecture_inspection_and_comparison():
    model_a = SimpleModel()
    model_b = SimpleModel()
    model_c = AlternateModel()

    info_a = ModelSerializer.inspect_model_architecture(model_a)
    info_b = ModelSerializer.inspect_model_architecture(model_b)
    info_c = ModelSerializer.inspect_model_architecture(model_c)

    assert info_a["model_type"] == "pytorch"
    assert "architecture_hash" in info_a

    compatible, details = ModelSerializer.compare_architecture_info(info_a, info_b)
    assert compatible
    assert details["mismatches"] == []

    incompatible, diff = ModelSerializer.compare_architecture_info(info_a, info_c)
    assert not incompatible
    assert diff["mismatches"]
