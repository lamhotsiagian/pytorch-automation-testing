from src.train import train_model


def test_training_reduces_loss():
    _, initial_loss, final_loss = train_model(
        n_samples=300,
        epochs=40,
        lr=0.05,
        device="cpu"
    )

    # Training should improve the loss
    assert final_loss < initial_loss, (
        f"Final loss ({final_loss:.4f}) should be less than initial loss ({initial_loss:.4f})"
    )

    # Optional: ensure it's not NAN or INF
    assert final_loss == final_loss  # not NaN
