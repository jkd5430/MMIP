import os

# 指定實體 GPU 4；這行必須放在匯入 TensorFlow 之前。
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse


TASKS = (
    "q2_plain_cnn",
    "q2_plain_cnn_2",
    "q2_lenet5",
    "q2_lenet5_2",
    "q3_plain_cnn_aug",
    "q3_lenet5_aug",
    "top3",
    "gradcam",
)


def run_task(task):
    if task == "q2_plain_cnn":
        from code.q2_plain_cnn import main
    elif task == "q2_plain_cnn_2":
        from code.q2_plain_cnn_2 import main
    elif task == "q2_lenet5":
        from code.q2_lenet5 import main
    elif task == "q2_lenet5_2":
        from code.q2_lenet5_2 import main
    elif task == "q3_plain_cnn_aug":
        from code.q3_plain_cnn_aug import main
    elif task == "q3_lenet5_aug":
        from code.q3_lenet5_aug import main
    elif task == "top3":
        from code.analyze_top3 import main
    else:
        from code.gradcam_lenet5_aug import main

    print(f"\n===== Running {task} on physical GPU 4 =====")
    main()


def main():
    parser = argparse.ArgumentParser(
        description="Run the OrganAMNIST Q2/Q3 experiments on physical GPU 4."
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="all",
        choices=(*TASKS, "q2_2", "all"),
        help="Task to run. Default: all",
    )
    args = parser.parse_args()

    if args.task == "all":
        selected = TASKS
    elif args.task == "q2_2":
        selected = ("q2_plain_cnn_2", "q2_lenet5_2")
    else:
        selected = (args.task,)
    for task in selected:
        run_task(task)


if __name__ == "__main__":
    main()
