from .common import run_experiment


def main():
    run_experiment("q2_lenet5", "lenet5", augmentation=False)
