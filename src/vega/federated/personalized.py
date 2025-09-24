"""
personalized.py: Personalized Federated Learning (FedPer, pFedMe) implementation for Vega2.0

Implements local adaptation and meta-learning for personalized federated learning.
"""

import copy
from typing import Any, Dict, Callable

import torch
from torch import nn, optim


class FedPerClient:
    """
    Federated Personalization Client (FedPer):
    - Freezes global backbone, adapts local head.
    """

    def __init__(
        self,
        model: nn.Module,
        backbone_layers: list,
        head_layers: list,
        lr: float = 0.01,
    ):
        self.model = model
        self.backbone_layers = backbone_layers
        self.head_layers = head_layers
        self.lr = lr
        self._freeze_backbone()
        self.optimizer = optim.SGD(self._head_parameters(), lr=self.lr)

    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in self.backbone_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _head_parameters(self):
        return [
            p
            for n, p in self.model.named_parameters()
            if any(layer in n for layer in self.head_layers)
        ]

    def local_adapt(self, data_loader, loss_fn: Callable):
        self.model.train()
        for x, y in data_loader:
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = loss_fn(output, y)
            loss.backward()
            self.optimizer.step()


class pFedMeClient:
    """
    Personalized Federated Meta-Learning (pFedMe):
    - Alternates between global and personalized model updates.
    """

    def __init__(
        self, model: nn.Module, lr: float = 0.01, lam: float = 15.0, K: int = 5
    ):
        self.model = model
        self.personalized_model = copy.deepcopy(model)
        self.lr = lr
        self.lam = lam
        self.K = K
        self.optimizer = optim.SGD(self.personalized_model.parameters(), lr=self.lr)

    def local_update(self, data_loader, loss_fn: Callable):
        self.personalized_model.train()
        for _ in range(self.K):
            for x, y in data_loader:
                self.optimizer.zero_grad()
                output = self.personalized_model(x)
                loss = loss_fn(output, y)
                # pFedMe regularization
                reg = 0.0
                for p, g in zip(
                    self.personalized_model.parameters(), self.model.parameters()
                ):
                    reg += torch.sum((p - g.detach()) ** 2)
                loss = loss + (self.lam / 2) * reg
                loss.backward()
                self.optimizer.step()

    def get_personalized_weights(self):
        return {
            k: v.cpu().clone() for k, v in self.personalized_model.state_dict().items()
        }

    def set_global_weights(self, global_state: Dict[str, Any]):
        self.model.load_state_dict(global_state)
        # Optionally, update personalized model as well
        # self.personalized_model.load_state_dict(global_state)


# Example usage (to be integrated with federated pipeline):
# client = FedPerClient(model, backbone_layers=["features"], head_layers=["classifier"])
# client.local_adapt(train_loader, nn.CrossEntropyLoss())
#
# pfedme = pFedMeClient(model)
# pfedme.local_update(train_loader, nn.CrossEntropyLoss())
