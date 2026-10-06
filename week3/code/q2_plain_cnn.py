from .common import run_experiment


def main():
    run_experiment("q2_plain_cnn", "plain_cnn", augmentation=False)
