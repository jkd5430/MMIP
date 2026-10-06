from .common import run_experiment


def main():
    run_experiment("q3_plain_cnn_aug", "plain_cnn", augmentation=True)
