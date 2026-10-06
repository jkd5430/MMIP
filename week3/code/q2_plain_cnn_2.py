from .common import run_experiment


def main():
    run_experiment(
        "q2_plain_cnn_2",
        "plain_cnn",
        augmentation=False,
        batch_size=64,
        learning_rate=5e-4,
        epochs=60,
        patience=10,
        focal_gamma=4.0,
    )
