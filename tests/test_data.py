from src.data import generate_linear_data


def test_generate_linear_data_shapes():
    n_samples = 200
    X, y = generate_linear_data(n_samples)

    assert X.shape == (n_samples, 1)
    assert y.shape == (n_samples, 1)
