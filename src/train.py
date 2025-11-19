import torch
from torch import nn, optim

from .data import generate_linear_data
from .model import SimpleRegressor


def train_model(
    n_samples: int = 500,
    epochs: int = 50,
    lr: float = 0.01,
    device: str = "cpu"
):
    """
    Train SimpleRegressor on synthetic linear data.

    Returns:
        model: trained PyTorch model
        initial_loss (float)
        final_loss (float)
    """
    torch.manual_seed(42)

    # 1. Generate data
    X, y = generate_linear_data(n_samples=n_samples)
    X, y = X.to(device), y.to(device)

    # 2. Build model, loss, optimizer
    model = SimpleRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 3. Initial loss (before training)
    model.train()
    with torch.no_grad():
        initial_loss = criterion(model(X), y).item()

    # 4. Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

    # 5. Final loss (after training)
    with torch.no_grad():
        final_loss = criterion(model(X), y).item()

    return model, initial_loss, final_loss


if __name__ == "__main__":
    model, initial_loss, final_loss = train_model()
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
