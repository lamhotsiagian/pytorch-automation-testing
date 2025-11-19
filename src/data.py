import torch


def generate_linear_data(n_samples: int = 1000):
    """
    Generate data for y = 2x + 1 + noise
    Returns:
        X: shape (n_samples, 1)
        y: shape (n_samples, 1)
    """
    torch.manual_seed(42)
    X = torch.rand(n_samples, 1) * 10  # values between 0 and 10
    noise = torch.randn(n_samples, 1) * 0.5
    y = 2 * X + 1 + noise
    return X, y
