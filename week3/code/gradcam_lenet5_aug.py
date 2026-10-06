import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from .common import RESULTS_ROOT, load_data, setup_gpu


RUN_DIR = RESULTS_ROOT / "q3_lenet5_aug"
OUTPUT_DIR = RUN_DIR / "gradcam"


def select_examples():
    with (RUN_DIR / "test_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    correct = [row for row in rows if row["correct"] == "1"][:5]
    wrong = [row for row in rows if row["correct"] == "0"][:5]
    return correct + wrong


def gradcam(model, image, target_class):
    # conv3 is 1x1, so conv2 is used to retain spatial information.
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer("conv2").output, model.output]
    )
    with tf.GradientTape() as tape:
        feature_maps, logits = grad_model(image, training=False)
        target_score = logits[:, target_class]
    gradients = tape.gradient(target_score, feature_maps)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return tf.image.resize(heatmap[..., None], (28, 28)).numpy()[..., 0]


def main():
    tf.keras.backend.clear_session()
    setup_gpu()
    data = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(str(RUN_DIR / "best.h5"), compile=False)
    examples = select_examples()

    summary = []
    for number, row in enumerate(examples):
        test_index = int(row["test_index"])
        true_index = int(row["true_index"])
        pred_index = int(row["pred_index"])
        gray = data["test_x"][test_index, ..., 0]
        model_input = gray[None, ..., None].astype(np.float32) / 255.0
        heatmap = gradcam(model, model_input, pred_index)

        color = plt.get_cmap("jet")(heatmap)[..., :3]
        original_rgb = np.repeat(gray[..., None], 3, axis=-1) / 255.0
        overlay = np.clip(0.6 * original_rgb + 0.4 * color, 0, 1)

        Image.fromarray(np.uint8(heatmap * 255)).save(
            OUTPUT_DIR / f"{number:02d}_heatmap.png"
        )
        Image.fromarray(np.uint8(overlay * 255)).save(
            OUTPUT_DIR / f"{number:02d}_overlay.png"
        )

        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title(f"True: {data['class_names'][true_index]}")
        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(f"Pred: {data['class_names'][pred_index]}")
        axes[2].imshow(overlay)
        axes[2].set_title(f"p={float(row['confidence']):.3f}")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{number:02d}_panel.png", dpi=180)
        plt.close(fig)

        summary.append(
            [
                test_index,
                data["class_names"][true_index],
                data["class_names"][pred_index],
                row["confidence"],
                row["correct"],
                f"{number:02d}_panel.png",
            ]
        )

    with (OUTPUT_DIR / "gradcam_examples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["test_index", "true_class", "pred_class", "confidence", "correct", "image"]
        )
        writer.writerows(summary)

    # Q3: visualize two learned kernels from the first convolution.
    kernels = model.get_layer("conv1").get_weights()[0]
    limit = max(float(np.abs(kernels[..., :2]).max()), 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    for index, axis in enumerate(axes):
        axis.imshow(
            kernels[:, :, 0, index], cmap="coolwarm", vmin=-limit, vmax=limit
        )
        axis.set_title(f"conv1 kernel {index}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "two_kernels.png", dpi=180)
    plt.close(fig)
    print("Grad-CAM results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
