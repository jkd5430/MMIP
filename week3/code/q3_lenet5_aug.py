from .common import run_experiment


def main():
    run_experiment("q3_lenet5_aug", "lenet5", augmentation=True)
