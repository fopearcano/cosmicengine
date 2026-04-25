"""CPU training loop for :class:`SpacetimeMLP`."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cosmic_engine.ai.training.dataset import SpacetimeDataset
from cosmic_engine.ai.training.model import SpacetimeMLP
from cosmic_engine.physics.nbody import GRAVITATIONAL_CONSTANT


def train_model(
    dataset: SpacetimeDataset,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    *,
    verbose: bool = True,
    seed: int = 0,
) -> tuple[SpacetimeMLP, list[float]]:
    """Train a fresh :class:`SpacetimeMLP` on ``dataset``.

    Returns the trained model and the list of per-epoch mean training
    losses (for tests / plotting). Targets and inputs come from
    :meth:`SpacetimeDataset.generate`. Adam optimizer, MSE loss.
    """
    torch.manual_seed(seed)
    x_np, y_np = dataset.generate()
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    # Derive normalization scales from the physical regime so the MLP's
    # input and output stay in O(1) magnitude regardless of the units
    # used at the call site.
    r_char = float(
        np.sqrt(dataset.radius_range[0] * dataset.radius_range[1])
    )
    a_char = max(
        2.0 * GRAVITATIONAL_CONSTANT * dataset.mass_kg / (r_char * r_char),
        1.0e-30,
    )
    model = SpacetimeMLP(r_char=r_char, a_char=a_char)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    n = x.shape[0]
    losses: list[float] = []
    for epoch in range(epochs):
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        epoch_count = 0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            xb = x[indices]
            yb = y[indices]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_count += 1
        mean_loss = epoch_loss / max(epoch_count, 1)
        losses.append(mean_loss)
        if verbose:
            print(f"epoch {epoch + 1:>3} / {epochs}  mean_loss = {mean_loss:.6e}")
    if verbose:
        print(f"final mean loss: {losses[-1]:.6e}")
    return model, losses
