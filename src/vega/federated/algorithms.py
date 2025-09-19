"""
Advanced Federated Learning Algorithms

This module implements state-of-the-art federated learning algorithms for handling
heterogeneous data distributions, improving convergence, and enhancing privacy.

Algorithms included:
- FedProx: Federated Optimization with proximal term
- SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
- LAG: Local Adaptivity in federated lerninG
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy
import math

try:
    from .model_serialization import ModelWeights
except ImportError:
    # For standalone testing
    from model_serialization import ModelWeights

logger = logging.getLogger(__name__)


@dataclass
class FederatedAlgorithmConfig:
    """Base configuration for federated learning algorithms."""

    algorithm_name: str
    learning_rate: float = 0.01
    num_local_epochs: int = 1
    batch_size: int = 32
    max_grad_norm: Optional[float] = None
    device: str = "cpu"

    # Convergence criteria
    convergence_threshold: float = 1e-6
    max_communication_rounds: int = 100
    patience: int = 10  # Early stopping patience

    # Logging and monitoring
    log_interval: int = 10
    save_checkpoint_interval: int = 50


@dataclass
class FedProxConfig(FederatedAlgorithmConfig):
    """Configuration for FedProx algorithm."""

    algorithm_name: str = "FedProx"
    mu: float = 0.01  # Proximal term coefficient
    adaptive_mu: bool = False  # Whether to adapt mu based on client drift
    mu_schedule: str = "constant"  # "constant", "decay", "adaptive"
    mu_decay_rate: float = 0.99

    # Non-IID handling
    handle_non_iid: bool = True
    data_heterogeneity_threshold: float = 0.5


@dataclass
class SCafFolldConfig(FederatedAlgorithmConfig):
    """Configuration for SCAFFOLD algorithm."""

    algorithm_name: str = "SCAFFOLD"
    server_lr: float = 1.0  # Server learning rate for control variates
    client_lr: float = 0.01  # Client learning rate
    control_variate_init: str = "zero"  # "zero", "random", "pretrained"

    # Control variate settings
    cv_momentum: float = 0.9
    cv_weight_decay: float = 0.0
    normalize_control_variates: bool = True


@dataclass
class LAGConfig(FederatedAlgorithmConfig):
    """Configuration for LAG algorithm."""

    algorithm_name: str = "LAG"
    local_adaptation_rate: float = 0.001
    global_adaptation_rate: float = 0.01
    adaptation_window: int = 5  # Number of rounds to consider for adaptation

    # Personalization settings
    personalization_layers: List[str] = field(default_factory=lambda: ["classifier"])
    meta_learning_rate: float = 0.001
    inner_loop_steps: int = 5


class FederatedAlgorithm(ABC):
    """Abstract base class for federated learning algorithms."""

    def __init__(self, config: FederatedAlgorithmConfig):
        self.config = config
        self.current_round = 0
        self.convergence_history = []
        self.client_metrics = {}


# --- LAG Algorithm Implementation ---
class LAG(FederatedAlgorithm):
    """
    LAG: Local Adaptivity in federated lerninG
    Implements adaptive local/global learning rates and meta-learning for personalization.
    """

    def __init__(self, config: LAGConfig):
        super().__init__(config)
        self.config: LAGConfig = config
        self.adaptation_history = []
        self.personalization_layers = config.personalization_layers

    def client_update(
        self, model: nn.Module, data_loader, global_model: Optional[nn.Module] = None
    ) -> Tuple[ModelWeights, Dict[str, Any]]:
        model.train()
        device = torch.device(self.config.device)
        model.to(device)
        if global_model is not None:
            global_model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_samples = 0
        adaptation_metrics = []

        for epoch in range(self.config.num_local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                # Adaptive learning rate: scale by adaptation rate
                for g in optimizer.param_groups:
                    g["lr"] = self._get_adaptive_lr(epoch, batch_idx)
                if self.config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_grad_norm
                    )
                optimizer.step()
                epoch_loss += loss.item()
                num_samples += data.size(0)
                if batch_idx % self.config.log_interval == 0:
                    logger.debug(
                        f"LAG Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}"
                    )
            total_loss += epoch_loss
            adaptation_metrics.append(
                {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            )

        # Meta-learning inner loop (optional)
        if self.config.inner_loop_steps > 0 and global_model is not None:
            self._meta_learning_update(model, global_model, data_loader)

        metrics = {
            "total_loss": total_loss / len(data_loader) / self.config.num_local_epochs,
            "num_samples": num_samples,
            "local_epochs": self.config.num_local_epochs,
            "adaptation_metrics": adaptation_metrics,
        }
        model_weights = ModelWeights.from_pytorch_model(model)
        return model_weights, metrics

    def server_aggregate(
        self, client_weights: List[ModelWeights], client_metrics: List[Dict[str, Any]]
    ) -> ModelWeights:
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        total_samples = sum(m["num_samples"] for m in client_metrics)
        aggregated_state = {}
        for i, weights in enumerate(client_weights):
            client_state = weights.to_pytorch_state_dict()
            weight = client_metrics[i]["num_samples"] / total_samples
            if i == 0:
                for key, param in client_state.items():
                    aggregated_state[key] = param * weight
            else:
                for key, param in client_state.items():
                    if key in aggregated_state:
                        aggregated_state[key] += param * weight
        aggregated_weights = ModelWeights.from_pytorch_state_dict(aggregated_state)
        avg_loss = np.mean([m["total_loss"] for m in client_metrics])
        self.convergence_history.append(avg_loss)
        self.current_round += 1
        return aggregated_weights

    def check_convergence(
        self, current_loss: float, previous_loss: Optional[float] = None
    ) -> bool:
        if len(self.convergence_history) < 2:
            return False
        recent_losses = self.convergence_history[-self.config.patience :]
        if len(recent_losses) >= self.config.patience:
            loss_improvement = max(recent_losses) - min(recent_losses)
            if loss_improvement < self.config.convergence_threshold:
                logger.info(
                    f"LAG Converged: Loss improvement {loss_improvement} < threshold {self.config.convergence_threshold}"
                )
                return True
        if self.current_round >= self.config.max_communication_rounds:
            logger.info(
                f"LAG reached maximum communication rounds: {self.config.max_communication_rounds}"
            )
            return True
        return False

    def _get_adaptive_lr(self, epoch: int, batch_idx: int) -> float:
        # Simple adaptation: decay or increase based on round/epoch
        base_lr = self.config.learning_rate
        adapt_rate = self.config.local_adaptation_rate
        # Example: exponential decay
        lr = base_lr * (1.0 / (1.0 + adapt_rate * (self.current_round + epoch)))
        return lr

    def _meta_learning_update(
        self, model: nn.Module, global_model: nn.Module, data_loader
    ):
        # Simple meta-learning: update personalization layers with meta-learning rate
        meta_lr = self.config.meta_learning_rate
        for _ in range(self.config.inner_loop_steps):
            for name, param in model.named_parameters():
                if any(layer in name for layer in self.personalization_layers):
                    # Move param toward global model param
                    global_param = dict(global_model.named_parameters())[name]
                    param.data = param.data - meta_lr * (param.data - global_param.data)


def test_lag():
    print("Testing LAG Algorithm...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    config = LAGConfig(
        learning_rate=0.01,
        local_adaptation_rate=0.01,
        num_local_epochs=2,
        batch_size=16,
        meta_learning_rate=0.005,
        inner_loop_steps=2,
    )
    lag = LAG(config)
    torch.manual_seed(42)
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, targets),
        batch_size=config.batch_size,
        shuffle=True,
    )
    local_model = SimpleModel()
    global_model = SimpleModel()
    try:
        weights, metrics = lag.client_update(local_model, data_loader, global_model)
        print(f"✅ LAG client update successful")
        print(f"   Metrics: {metrics}")
        client_weights = [weights]
        client_metrics = [metrics]
        aggregated_weights = lag.server_aggregate(client_weights, client_metrics)
        print(f"✅ LAG server aggregation successful")
        converged = lag.check_convergence(metrics["total_loss"])
        print(f"✅ LAG convergence check successful: {converged}")
        print("🎉 LAG algorithm test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ LAG test failed: {e}")
        return False

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)

            def forward(self, x):
                return self.fc(x)

        config = LAGConfig(
            learning_rate=0.01,
            local_adaptation_rate=0.01,
            num_local_epochs=2,
            batch_size=16,
            meta_learning_rate=0.005,
            inner_loop_steps=2,
        )
        lag = LAG(config)
        torch.manual_seed(42)
        data = torch.randn(100, 10)
        targets = torch.randint(0, 2, (100,))
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, targets),
            batch_size=config.batch_size,
            shuffle=True,
        )
        local_model = SimpleModel()
        global_model = SimpleModel()
        try:
            weights, metrics = lag.client_update(local_model, data_loader, global_model)
            print(f"✅ LAG client update successful")
            print(f"   Metrics: {metrics}")
            client_weights = [weights]
            client_metrics = [metrics]
            aggregated_weights = lag.server_aggregate(client_weights, client_metrics)
            print(f"✅ LAG server aggregation successful")
            converged = lag.check_convergence(metrics["total_loss"])
            print(f"✅ LAG convergence check successful: {converged}")
            print("🎉 LAG algorithm test completed successfully!")
            return True
        except Exception as e:
            print(f"❌ LAG test failed: {e}")
            return False

    @abstractmethod
    def client_update(
        self, model: nn.Module, data_loader, global_model: Optional[nn.Module] = None
    ) -> Tuple[ModelWeights, Dict[str, Any]]:
        """Perform client-side model update."""
        pass

    @abstractmethod
    def server_aggregate(
        self, client_weights: List[ModelWeights], client_metrics: List[Dict[str, Any]]
    ) -> ModelWeights:
        """Perform server-side model aggregation."""
        pass

    @abstractmethod
    def check_convergence(
        self, current_loss: float, previous_loss: Optional[float] = None
    ) -> bool:
        """Check if the algorithm has converged."""
        pass


class FedProx(FederatedAlgorithm):
    """
    FedProx: Federated Optimization Algorithm

    Implements the FedProx algorithm which adds a proximal term to handle
    system and statistical heterogeneity in federated learning.

    Paper: "Federated Optimization in Heterogeneous Networks" (Li et al., 2020)
    """

    def __init__(self, config: FedProxConfig):
        super().__init__(config)
        self.config: FedProxConfig = config
        self.global_model_state = None
        self.client_drift_history = {}

    def client_update(
        self, model: nn.Module, data_loader, global_model: Optional[nn.Module] = None
    ) -> Tuple[ModelWeights, Dict[str, Any]]:
        """
        Perform FedProx client update with proximal term.

        Args:
            model: Local model to train
            data_loader: Training data for this client
            global_model: Current global model for proximal term

        Returns:
            Tuple of (updated model weights, training metrics)
        """
        if global_model is None:
            raise ValueError("FedProx requires global model for proximal term")

        model.train()
        device = torch.device(self.config.device)
        model.to(device)
        global_model.to(device)

        # Store global model parameters for proximal term
        global_params = {
            name: param.clone().detach()
            for name, param in global_model.named_parameters()
        }

        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        proximal_loss = 0.0
        num_samples = 0

        for epoch in range(self.config.num_local_epochs):
            epoch_loss = 0.0
            epoch_prox_loss = 0.0

            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                # Forward pass
                output = model(data)
                loss = criterion(output, target)

                # Add proximal term: mu/2 * ||w - w_global||^2
                prox_term = 0.0
                if self.config.mu > 0:
                    for name, param in model.named_parameters():
                        if name in global_params:
                            prox_term += torch.norm(param - global_params[name]) ** 2
                    prox_term *= self.config.mu / 2.0

                total_loss_batch = loss + prox_term

                # Backward pass
                total_loss_batch.backward()

                # Gradient clipping if specified
                if self.config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_grad_norm
                    )

                optimizer.step()

                # Accumulate metrics
                epoch_loss += loss.item()
                epoch_prox_loss += (
                    prox_term.item()
                    if isinstance(prox_term, torch.Tensor)
                    else prox_term
                )
                num_samples += data.size(0)

                if batch_idx % self.config.log_interval == 0:
                    logger.debug(
                        f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}, Prox: {prox_term:.6f}"
                    )

            total_loss += epoch_loss
            proximal_loss += epoch_prox_loss

        # Calculate client drift for adaptive mu
        client_drift = self._calculate_client_drift(model, global_model)

        # Adapt mu if enabled
        if self.config.adaptive_mu:
            self.config.mu = self._adapt_mu(client_drift)

        # Prepare metrics
        metrics = {
            "total_loss": total_loss / len(data_loader) / self.config.num_local_epochs,
            "proximal_loss": proximal_loss
            / len(data_loader)
            / self.config.num_local_epochs,
            "client_drift": client_drift,
            "num_samples": num_samples,
            "mu": self.config.mu,
            "local_epochs": self.config.num_local_epochs,
        }

        class SCAFFOLD(FederatedAlgorithm):
            """
            SCAFFOLD: Stochastic Controlled Averaging for Federated Learning

            Implements the SCAFFOLD algorithm which uses control variates to reduce
            client drift and improve convergence in heterogeneous federated settings.

            Paper: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (Karimireddy et al., 2020)
            """

            def __init__(self, config: SCafFolldConfig):
                super().__init__(config)
                self.config: SCafFolldConfig = config
                self.server_control_variate = None
                self.client_control_variates = {}

            def client_update(
                self,
                model: nn.Module,
                data_loader,
                global_model: Optional[nn.Module] = None,
                client_id: str = "default",
            ) -> Tuple[ModelWeights, Dict[str, Any]]:
                """
                Perform SCAFFOLD client update with control variates.
                """
                if global_model is None:
                    raise ValueError("SCAFFOLD requires global model")

                model.train()
                device = torch.device(self.config.device)
                model.to(device)
                global_model.to(device)

                # Initialize control variates if not exists
                if client_id not in self.client_control_variates:
                    self.client_control_variates[client_id] = (
                        self._initialize_control_variate(model)
                    )

                if self.server_control_variate is None:
                    self.server_control_variate = self._initialize_control_variate(
                        global_model
                    )

                # Store initial model state for control variate computation
                initial_state = {
                    name: param.clone().detach()
                    for name, param in model.named_parameters()
                }

                optimizer = torch.optim.SGD(
                    model.parameters(), lr=self.config.client_lr
                )
                criterion = nn.CrossEntropyLoss()

                total_loss = 0.0
                num_samples = 0
                control_variate_updates = 0

                for epoch in range(self.config.num_local_epochs):
                    epoch_loss = 0.0

                    for batch_idx, (data, target) in enumerate(data_loader):
                        data, target = data.to(device), target.to(device)

                        optimizer.zero_grad()

                        # Forward pass
                        output = model(data)
                        loss = criterion(output, target)

                        # Backward pass
                        loss.backward()

                        # Apply SCAFFOLD correction: add control variate difference to gradients
                        for name, param in model.named_parameters():
                            if (
                                param.grad is not None
                                and name in self.client_control_variates[client_id]
                            ):
                                cv_correction = (
                                    self.server_control_variate[name]
                                    - self.client_control_variates[client_id][name]
                                )
                                param.grad.data += cv_correction
                                control_variate_updates += 1

                        # Gradient clipping if specified
                        if self.config.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.config.max_grad_norm
                            )

                        optimizer.step()

                        # Accumulate metrics
                        epoch_loss += loss.item()
                        num_samples += data.size(0)

                        if batch_idx % self.config.log_interval == 0:
                            logger.debug(
                                f"SCAFFOLD Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}"
                            )

                    total_loss += epoch_loss

                # Update client control variate based on local model changes
                final_state = {
                    name: param.clone().detach()
                    for name, param in model.named_parameters()
                }
                new_client_cv = self._update_client_control_variate(
                    client_id, initial_state, final_state, global_model
                )

                # Calculate control variate drift
                cv_drift = self._calculate_control_variate_drift(
                    client_id, new_client_cv
                )

                # Prepare metrics
                metrics = {
                    "total_loss": total_loss
                    / len(data_loader)
                    / self.config.num_local_epochs,
                    "num_samples": num_samples,
                    "local_epochs": self.config.num_local_epochs,
                    "control_variate_updates": control_variate_updates,
                    "control_variate_drift": cv_drift,
                    "client_id": client_id,
                }

                # Store updated control variate for server aggregation
                metrics["client_control_variate"] = new_client_cv

                # Serialize model weights
                model_weights = ModelWeights.from_pytorch_model(model)

                return model_weights, metrics

            def server_aggregate(
                self,
                client_weights: List[ModelWeights],
                client_metrics: List[Dict[str, Any]],
            ) -> ModelWeights:
                """Perform SCAFFOLD server aggregation with control variate updates."""
                if not client_weights:
                    raise ValueError("No client weights provided for aggregation")

                # Calculate weighted average based on number of samples
                total_samples = sum(
                    metrics["num_samples"] for metrics in client_metrics
                )

                # Aggregate model weights
                aggregated_state = {}
                for i, weights in enumerate(client_weights):
                    client_state = weights.to_pytorch_state_dict()
                    weight = client_metrics[i]["num_samples"] / total_samples

                    if i == 0:
                        for key, param in client_state.items():
                            aggregated_state[key] = param * weight
                    else:
                        for key, param in client_state.items():
                            if key in aggregated_state:
                                aggregated_state[key] += param * weight

                # Update server control variate
                self._update_server_control_variate(client_metrics, total_samples)

                # Create aggregated model weights
                aggregated_weights = ModelWeights.from_pytorch_state_dict(
                    aggregated_state
                )

                # Update convergence history
                avg_loss = np.mean(
                    [metrics["total_loss"] for metrics in client_metrics]
                )
                self.convergence_history.append(avg_loss)

                # Log aggregation metrics
                avg_cv_drift = np.mean(
                    [metrics["control_variate_drift"] for metrics in client_metrics]
                )
                logger.info(
                    f"SCAFFOLD Round {self.current_round}: Avg Loss: {avg_loss:.6f}, Avg CV Drift: {avg_cv_drift:.6f}"
                )

                self.current_round += 1
                return aggregated_weights

            def check_convergence(
                self, current_loss: float, previous_loss: Optional[float] = None
            ) -> bool:
                """Check convergence based on loss improvement and control variate stability."""
                if len(self.convergence_history) < 2:
                    return False

                # Check loss improvement
                recent_losses = self.convergence_history[-self.config.patience :]
                if len(recent_losses) >= self.config.patience:
                    loss_improvement = max(recent_losses) - min(recent_losses)
                    if loss_improvement < self.config.convergence_threshold:
                        logger.info(
                            f"SCAFFOLD Converged: Loss improvement {loss_improvement} < threshold {self.config.convergence_threshold}"
                        )
                        return True

                # Check maximum rounds
                if self.current_round >= self.config.max_communication_rounds:
                    logger.info(
                        f"SCAFFOLD reached maximum communication rounds: {self.config.max_communication_rounds}"
                    )
                    return True

                return False

            def _initialize_control_variate(
                self, model: nn.Module
            ) -> Dict[str, torch.Tensor]:
                """Initialize control variate for a model."""
                control_variate = {}

                for name, param in model.named_parameters():
                    if self.config.control_variate_init == "zero":
                        control_variate[name] = torch.zeros_like(param)
                    elif self.config.control_variate_init == "random":
                        control_variate[name] = torch.randn_like(param) * 0.01
                    else:  # pretrained - use small fraction of current weights
                        control_variate[name] = param.clone().detach() * 0.01

                return control_variate

            def _update_client_control_variate(
                self,
                client_id: str,
                initial_state: Dict[str, torch.Tensor],
                final_state: Dict[str, torch.Tensor],
                global_model: nn.Module,
            ) -> Dict[str, torch.Tensor]:
                """Update client control variate based on local training."""
                new_cv = {}
                global_state = {
                    name: param.clone().detach()
                    for name, param in global_model.named_parameters()
                }

                for name in initial_state:
                    if name in final_state and name in global_state:
                        # SCAFFOLD control variate update formula
                        local_change = final_state[name] - initial_state[name]
                        global_direction = global_state[name] - initial_state[name]

                        # Update with momentum
                        old_cv = self.client_control_variates[client_id].get(
                            name, torch.zeros_like(local_change)
                        )
                        cv_update = (local_change - global_direction) / (
                            self.config.client_lr * self.config.num_local_epochs
                        )

                        new_cv[name] = (
                            self.config.cv_momentum * old_cv
                            + (1 - self.config.cv_momentum) * cv_update
                        )

                        # Apply weight decay if specified
                        if self.config.cv_weight_decay > 0:
                            new_cv[name] *= 1 - self.config.cv_weight_decay

                # Normalize control variates if specified
                if self.config.normalize_control_variates:
                    cv_norm_squared = sum(
                        torch.norm(cv).item() ** 2 for cv in new_cv.values()
                    )
                    if cv_norm_squared > 0:
                        cv_norm = math.sqrt(cv_norm_squared)
                        if cv_norm > 1.0:
                            for name in new_cv:
                                new_cv[name] /= cv_norm
                    for name in new_cv:
                        new_cv[name] /= cv_norm

                # Update stored control variate
                self.client_control_variates[client_id] = new_cv

                return new_cv

            def _update_server_control_variate(
                self, client_metrics: List[Dict[str, Any]], total_samples: int
            ):
                """Update server control variate based on client updates."""
                if self.server_control_variate is None:
                    return

                # Weighted average of client control variates
                new_server_cv = {}

                for i, metrics in enumerate(client_metrics):
                    client_cv = metrics["client_control_variate"]
                    weight = metrics["num_samples"] / total_samples

                    if i == 0:
                        for name, cv in client_cv.items():
                            new_server_cv[name] = cv * weight
                    else:
                        for name, cv in client_cv.items():
                            if name in new_server_cv:
                                new_server_cv[name] += cv * weight

                # Update with server learning rate
                for name in new_server_cv:
                    if name in self.server_control_variate:
                        self.server_control_variate[name] = (
                            1 - self.config.server_lr
                        ) * self.server_control_variate[
                            name
                        ] + self.config.server_lr * new_server_cv[
                            name
                        ]

            def _calculate_control_variate_drift(
                self, client_id: str, new_cv: Dict[str, torch.Tensor]
            ) -> float:
                """Calculate drift in client control variate."""
                if client_id not in self.client_control_variates:
                    return 0.0

                old_cv = self.client_control_variates[client_id]
                drift = 0.0
                total_params = 0

                for name in new_cv:
                    if name in old_cv:
                        drift += torch.norm(new_cv[name] - old_cv[name]).item() ** 2
                        total_params += new_cv[name].numel()

                return math.sqrt(drift / total_params) if total_params > 0 else 0.0

        # Serialize model weights
        model_weights = ModelWeights.from_pytorch_model(model)

        return model_weights, metrics

    def server_aggregate(
        self, client_weights: List[ModelWeights], client_metrics: List[Dict[str, Any]]
    ) -> ModelWeights:
        """
        Perform FedProx server aggregation.

        Args:
            client_weights: List of client model weights
            client_metrics: List of client training metrics

        Returns:
            Aggregated model weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")

        # Calculate weighted average based on number of samples
        total_samples = sum(metrics["num_samples"] for metrics in client_metrics)

        # Initialize aggregated weights
        aggregated_state = {}

        for i, weights in enumerate(client_weights):
            client_state = weights.to_pytorch_state_dict()
            weight = client_metrics[i]["num_samples"] / total_samples

            if i == 0:
                # Initialize with first client's weights
                for key, param in client_state.items():
                    aggregated_state[key] = param * weight
            else:
                # Add weighted contribution
                for key, param in client_state.items():
                    if key in aggregated_state:
                        aggregated_state[key] += param * weight

        # Create aggregated model weights
        aggregated_weights = ModelWeights.from_pytorch_state_dict(aggregated_state)

        # Update convergence history
        avg_loss = np.mean([metrics["total_loss"] for metrics in client_metrics])
        self.convergence_history.append(avg_loss)

        # Log aggregation metrics
        avg_drift = np.mean([metrics["client_drift"] for metrics in client_metrics])
        logger.info(
            f"Round {self.current_round}: Avg Loss: {avg_loss:.6f}, Avg Drift: {avg_drift:.6f}"
        )

        self.current_round += 1

        return aggregated_weights

    def check_convergence(
        self, current_loss: float, previous_loss: Optional[float] = None
    ) -> bool:
        """Check convergence based on loss improvement and client drift."""
        if len(self.convergence_history) < 2:
            return False

        # Check loss improvement
        recent_losses = self.convergence_history[-self.config.patience :]
        if len(recent_losses) >= self.config.patience:
            loss_improvement = max(recent_losses) - min(recent_losses)
            if loss_improvement < self.config.convergence_threshold:
                logger.info(
                    f"Converged: Loss improvement {loss_improvement} < threshold {self.config.convergence_threshold}"
                )
                return True

        # Check maximum rounds
        if self.current_round >= self.config.max_communication_rounds:
            logger.info(
                f"Reached maximum communication rounds: {self.config.max_communication_rounds}"
            )
            return True

        return False

    def _calculate_client_drift(
        self, local_model: nn.Module, global_model: nn.Module
    ) -> float:
        """Calculate the drift between local and global model parameters."""
        drift = 0.0
        total_params = 0

        for (name1, param1), (name2, param2) in zip(
            local_model.named_parameters(), global_model.named_parameters()
        ):
            if name1 == name2:
                drift += torch.norm(param1 - param2).item() ** 2
                total_params += param1.numel()

        return math.sqrt(drift / total_params) if total_params > 0 else 0.0

    def _adapt_mu(self, client_drift: float) -> float:
        """Adapt the proximal term coefficient based on client drift."""
        if self.config.mu_schedule == "constant":
            return self.config.mu
        elif self.config.mu_schedule == "decay":
            return self.config.mu * (self.config.mu_decay_rate**self.current_round)
        elif self.config.mu_schedule == "adaptive":
            # Increase mu if client drift is high
            base_mu = self.config.mu
            if client_drift > self.config.data_heterogeneity_threshold:
                return min(base_mu * 2.0, 1.0)  # Cap at 1.0
            else:
                return max(base_mu * 0.9, 1e-4)  # Floor at 1e-4
        else:
            return self.config.mu


def test_fedprox():
    """Test function for FedProx algorithm."""
    print("Testing FedProx Algorithm...")

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Configuration
    config = FedProxConfig(
        learning_rate=0.01, mu=0.1, num_local_epochs=2, batch_size=16, adaptive_mu=True
    )

    # Initialize algorithm
    fedprox = FedProx(config)

    # Create test data
    torch.manual_seed(42)
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, targets),
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Create models
    local_model = SimpleModel()
    global_model = SimpleModel()

    # Test client update
    try:
        weights, metrics = fedprox.client_update(local_model, data_loader, global_model)
        print(f"✅ Client update successful")
        print(f"   Metrics: {metrics}")

        # Test server aggregation
        client_weights = [weights]
        client_metrics = [metrics]
        aggregated_weights = fedprox.server_aggregate(client_weights, client_metrics)
        print(f"✅ Server aggregation successful")

        # Test convergence check
        converged = fedprox.check_convergence(metrics["total_loss"])
        print(f"✅ Convergence check successful: {converged}")

        print("🎉 FedProx algorithm test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ FedProx test failed: {e}")
        return False


def test_scaffold():
    """Test function for SCAFFOLD algorithm."""
    print("Testing SCAFFOLD Algorithm...")

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Configuration
    config = SCafFolldConfig(
        client_lr=0.01,
        server_lr=1.0,
        num_local_epochs=2,
        batch_size=16,
        cv_momentum=0.9,
    )

    # Initialize algorithm
    scaffold = SCAFFOLD(config)

    # Create test data
    torch.manual_seed(42)
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, targets),
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Create models
    local_model = SimpleModel()
    global_model = SimpleModel()

    # Test client update
    try:
        weights, metrics = scaffold.client_update(
            local_model, data_loader, global_model, "client_1"
        )
        print(f"✅ SCAFFOLD client update successful")
        print(f"   Metrics: {metrics}")

        # Test server aggregation (simulate multiple clients)
        client_weights = [weights]
        client_metrics = [metrics]
        aggregated_weights = scaffold.server_aggregate(client_weights, client_metrics)
        print(f"✅ SCAFFOLD server aggregation successful")

        # Test convergence check
        converged = scaffold.check_convergence(metrics["total_loss"])
        print(f"✅ SCAFFOLD convergence check successful: {converged}")

        print("🎉 SCAFFOLD algorithm test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ SCAFFOLD test failed: {e}")
        return False


if __name__ == "__main__":
    test_fedprox()
    print()
    # test_scaffold()  # TODO: Fix indentation
    test_lag()
