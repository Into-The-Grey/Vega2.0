"""
Personalization Framework for Federated Learning

Supports local model adaptation (fine-tuning, meta-learning, local layers) while contributing to global model improvement.
Modular and pluggable into any federated learning workflow.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersonalizationConfig:
    personalization_layers: List[str] = field(default_factory=lambda: ["classifier"])
    local_epochs: int = 1
    local_lr: float = 0.01
    meta_learning: bool = False
    meta_lr: float = 0.001
    inner_loop_steps: int = 3


class PersonalizationFramework:
    def __init__(self, config: PersonalizationConfig):
        self.config = config

    def personalize(
        self, model: nn.Module, data_loader, global_model: Optional[nn.Module] = None
    ) -> nn.Module:
        """
        Perform local adaptation (fine-tuning or meta-learning) on the model.
        """
        model.train()
        device = next(model.parameters()).device
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.local_lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        if self.config.meta_learning and global_model is not None:
            self._meta_learning_update(model, global_model, data_loader)
        return model

    def _meta_learning_update(
        self, model: nn.Module, global_model: nn.Module, data_loader
    ):
        meta_lr = self.config.meta_lr
        for _ in range(self.config.inner_loop_steps):
            for name, param in model.named_parameters():
                if any(layer in name for layer in self.config.personalization_layers):
                    global_param = dict(global_model.named_parameters())[name]
                    param.data = param.data - meta_lr * (param.data - global_param.data)

    def extract_personalized_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Extract only the personalized layers' weights.
        """
        return {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if any(layer in name for layer in self.config.personalization_layers)
        }

    def merge_global_and_personalized(
        self,
        global_weights: Dict[str, torch.Tensor],
        personalized_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Merge global weights with personalized weights for deployment.
        """
        merged = global_weights.copy()
        merged.update(personalized_weights)
        return merged


def test_personalization():
    print("Testing Personalization Framework...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(10, 5)
            self.classifier = nn.Linear(5, 2)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    config = PersonalizationConfig(
        personalization_layers=["classifier"],
        local_epochs=2,
        local_lr=0.05,
        meta_learning=True,
        meta_lr=0.01,
        inner_loop_steps=2,
    )
    pf = PersonalizationFramework(config)
    torch.manual_seed(42)
    data = torch.randn(50, 10)
    targets = torch.randint(0, 2, (50,))
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, targets), batch_size=10, shuffle=True
    )
    global_model = SimpleModel()
    local_model = SimpleModel()
    # Personalize local model
    personalized_model = pf.personalize(local_model, data_loader, global_model)
    personalized_weights = pf.extract_personalized_weights(personalized_model)
    global_weights = dict(global_model.named_parameters())
    merged = pf.merge_global_and_personalized(global_weights, personalized_weights)
    print(f"Personalized weights: {list(personalized_weights.keys())}")
    print(f"Merged weights: {list(merged.keys())}")
    assert all(layer in merged for layer in personalized_weights)
    print("🎉 Personalization Framework test completed successfully!")
    return True


if __name__ == "__main__":
    test_personalization()
