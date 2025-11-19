import torch
from src.model import SimpleRegressor


def test_model_forward_shape():
    model = SimpleRegressor(input_dim=1, hidden_dim=8)
    x = torch.rand(10, 1)  # batch of 10

    y = model(x)

    assert y.shape == (10, 1), "Output shape must be (batch_size, 1)"
