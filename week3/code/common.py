import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from medmnist import INFO, OrganAMNIST
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATA_ROOT = PROJECT_ROOT / "data" / "medmnist"
RESULTS_ROOT = PROJECT_ROOT / "results"
CLASS_COUNT = 11
LATERAL_CLASSES = [1, 2, 4, 5, 7, 8]
_GPU_CONFIGURED = False


def setup_gpu():
    global _GPU_CONFIGURED
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow 找不到 GPU，請檢查 GPU 4 與 CUDA 環境。")
    if not _GPU_CONFIGURED:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        _GPU_CONFIGURED = True
    tf.keras.utils.set_random_seed(SEED)
    print("TensorFlow GPU =", gpus)


def load_data():
    """Official train -> stratified 80/20; official test stays untouched."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    train_set = OrganAMNIST(
        split="train", root=str(DATA_ROOT), size=28, download=True
    )
    test_set = OrganAMNIST(
        split="test", root=str(DATA_ROOT), size=28, download=True
    )

    all_train_x = np.asarray(train_set.imgs, dtype=np.uint8)[..., None]
    all_train_y = np.asarray(train_set.labels, dtype=np.int32).reshape(-1)
    test_x = np.asarray(test_set.imgs, dtype=np.uint8)[..., None]
    test_y = np.asarray(test_set.labels, dtype=np.int32).reshape(-1)

    all_indices = np.arange(len(all_train_y))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=SEED,
        stratify=all_train_y,
    )
    train_indices.sort()
    val_indices.sort()

    class_names = [INFO["organamnist"]["label"][str(i)] for i in range(CLASS_COUNT)]
    split_hash = hashlib.sha256(
        train_indices.tobytes() + val_indices.tobytes()
    ).hexdigest()

    return {
        "train_x": all_train_x[train_indices],
        "train_y": all_train_y[train_indices],
        "val_x": all_train_x[val_indices],
        "val_y": all_train_y[val_indices],
        "test_x": test_x,
        "test_y": test_y,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "class_names": class_names,
        "split_hash": split_hash,
    }


def make_dataset(images, labels, batch_size, training=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    if training and augment:
        flip = tf.keras.layers.RandomFlip("horizontal", seed=SEED)
        move_rotate = tf.keras.Sequential(
            [
                tf.keras.layers.RandomTranslation(
                    0.05, 0.05, fill_mode="nearest", seed=SEED + 1
                ),
                tf.keras.layers.RandomRotation(
                    10 / 360, fill_mode="nearest", seed=SEED + 2
                ),
            ]
        )

        def apply_augmentation(x, y):
            x = tf.cast(x, tf.float32)
            flipped = flip(x, training=True)
            is_lateral = tf.reduce_any(
                tf.equal(y[:, None], tf.constant(LATERAL_CLASSES)[None, :]), axis=1
            )
            # Left/right femur, kidney and lung are not horizontally flipped.
            x = tf.where(is_lateral[:, None, None, None], x, flipped)
            return move_rotate(x, training=True), y

        ds = ds.map(apply_augmentation, num_parallel_calls=1)

    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_plain_cnn():
    layers = tf.keras.layers
    inputs = layers.Input((28, 28, 1))
    x = inputs
    for number, filters in enumerate([32, 32, 64, 64], start=1):
        x = layers.Conv2D(
            filters, 3, padding="valid", activation="relu", name=f"conv{number}"
        )(x)
        x = layers.BatchNormalization(name=f"bn{number}")(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(64, 3, padding="valid", activation="relu", name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(CLASS_COUNT, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="plain_cnn")


def build_lenet5():
    """LeNet-5 adapted to 11 classes; zero padding keeps the 28x28 image unscaled."""
    layers = tf.keras.layers
    inputs = layers.Input((28, 28, 1))
    x = layers.ZeroPadding2D(2)(inputs)  # 28x28 -> 32x32, no resizing
    x = layers.Conv2D(6, 5, activation="tanh", name="conv1")(x)
    x = layers.AveragePooling2D(2)(x)
    x = layers.Conv2D(16, 5, activation="tanh", name="conv2")(x)
    x = layers.AveragePooling2D(2)(x)
    x = layers.Conv2D(120, 5, activation="tanh", name="conv3")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(84, activation="tanh")(x)
    outputs = layers.Dense(CLASS_COUNT, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="lenet5")


def save_learning_curves(history, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for axis, metric in zip(axes, ["loss", "top1", "top3", "top5"]):
        axis.plot(history.history[metric], label="train")
        axis.plot(history.history[f"val_{metric}"], label="validation")
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


@tf.keras.utils.register_keras_serializable(package="MMIP")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Sparse focal loss for logits. gamma=4 strongly emphasizes hard samples."""

    def __init__(self, gamma=4.0, name="sparse_categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = float(gamma)

    def call(self, y_true, logits):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=logits
        )
        probabilities = tf.nn.softmax(logits, axis=-1)
        true_probability = tf.gather(probabilities, y_true, batch_dims=1)
        focal_weight = tf.pow(1.0 - true_probability, self.gamma)
        return focal_weight * cross_entropy

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config


def save_hard_samples(
    probabilities, test_y, test_x, class_names, output_dir, focal_gamma
):
    ranking = np.argsort(-probabilities, axis=1)
    prediction = ranking[:, 0]
    true_rank = np.argmax(ranking == test_y[:, None], axis=1) + 1
    true_probability = probabilities[np.arange(len(test_y)), test_y]
    focal_score = -np.power(1.0 - true_probability, focal_gamma) * np.log(
        np.clip(true_probability, 1e-8, 1.0)
    )
    hard_order = np.argsort(-focal_score)

    with (output_dir / "hard_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "hard_rank",
                "test_index",
                "true_class",
                "pred_class",
                "true_probability",
                "pred_probability",
                "true_rank",
                "focal_loss",
                "correct",
            ]
        )
        for hard_rank, index in enumerate(hard_order, start=1):
            writer.writerow(
                [
                    hard_rank,
                    int(index),
                    class_names[test_y[index]],
                    class_names[prediction[index]],
                    float(true_probability[index]),
                    float(probabilities[index, prediction[index]]),
                    int(true_rank[index]),
                    float(focal_score[index]),
                    int(prediction[index] == test_y[index]),
                ]
            )

    shown = hard_order[:20]
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for axis, index in zip(axes.flat, shown):
        axis.imshow(test_x[index, ..., 0], cmap="gray")
        axis.set_title(
            f"{class_names[test_y[index]]} -> {class_names[prediction[index]]}\n"
            f"p(true)={true_probability[index]:.3f}, rank={true_rank[index]}",
            fontsize=8,
        )
        axis.axis("off")
    fig.suptitle(f"20 hardest test samples, focal gamma={focal_gamma:g}")
    fig.tight_layout()
    fig.savefig(output_dir / "hard_samples.png", dpi=180)
    plt.close(fig)

    top100 = hard_order[:100]
    return {
        "hard_top100_wrong": int(np.sum(prediction[top100] != test_y[top100])),
        "hard_top100_mean_true_probability": float(np.mean(true_probability[top100])),
    }


def save_test_results(
    model,
    test_ds,
    test_y,
    class_names,
    output_dir,
    test_x=None,
    focal_gamma=None,
):
    logits = model.predict(test_ds, verbose=1)
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    prediction = probabilities.argmax(axis=1)
    top5 = np.argsort(-probabilities, axis=1)[:, :5]
    top3 = top5[:, :3]

    top1_accuracy = float(np.mean(prediction == test_y))
    top3_accuracy = float(np.mean(np.any(top3 == test_y[:, None], axis=1)))
    top5_accuracy = float(np.mean(np.any(top5 == test_y[:, None], axis=1)))

    np.save(output_dir / "test_probabilities.npy", probabilities)
    np.save(output_dir / "test_labels.npy", test_y)

    hard_metrics = {}
    if focal_gamma is not None and test_x is not None:
        hard_metrics = save_hard_samples(
            probabilities,
            test_y,
            test_x,
            class_names,
            output_dir,
            focal_gamma,
        )

    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_index",
                "true_index",
                "true_class",
                "pred_index",
                "pred_class",
                "confidence",
                "correct",
                "top3_indices",
                "top3_classes",
                "top5_indices",
                "top5_classes",
            ]
        )
        for index in range(len(test_y)):
            writer.writerow(
                [
                    index,
                    int(test_y[index]),
                    class_names[test_y[index]],
                    int(prediction[index]),
                    class_names[prediction[index]],
                    float(probabilities[index, prediction[index]]),
                    int(prediction[index] == test_y[index]),
                    json.dumps(top3[index].tolist()),
                    json.dumps([class_names[i] for i in top3[index]]),
                    json.dumps(top5[index].tolist()),
                    json.dumps([class_names[i] for i in top5[index]]),
                ]
            )

    matrix = confusion_matrix(test_y, prediction, labels=np.arange(CLASS_COUNT))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")

    auc_values = []
    fig, axis = plt.subplots(figsize=(8, 7))
    with (output_dir / "per_class_auc.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "class_name", "auc"])
        for class_index, class_name in enumerate(class_names):
            binary_truth = (test_y == class_index).astype(np.int32)
            false_positive, true_positive, _ = roc_curve(
                binary_truth, probabilities[:, class_index]
            )
            class_auc = float(auc(false_positive, true_positive))
            auc_values.append(class_auc)
            writer.writerow([class_index, class_name, class_auc])
            axis.plot(
                false_positive,
                true_positive,
                label=f"{class_name} ({class_auc:.3f})",
            )
    macro_auc = float(np.mean(auc_values))
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"One-vs-rest ROC, Macro-AUC = {macro_auc:.4f}")
    axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    return {
        "test_top1": top1_accuracy,
        "test_top3": top3_accuracy,
        "test_top5": top5_accuracy,
        "test_macro_auc": macro_auc,
        **hard_metrics,
    }

def update_comparison():
    rows = []
    for metrics_file in sorted(RESULTS_ROOT.glob("*/metrics.json")):
        rows.append(json.loads(metrics_file.read_text(encoding="utf-8")))
    if not rows:
        return
    fields = [
        "run_name",
        "model",
        "augmentation",
        "loss",
        "focal_gamma",
        "learning_rate",
        "batch_size",
        "parameters",
        "best_epoch",
        "val_top1",
        "val_top3",
        "val_top5",
        "test_top1",
        "test_top3",
        "test_top5",
        "test_macro_auc",
    ]
    with (RESULTS_ROOT / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

def make_metrics():
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
    ]


def run_experiment(
    run_name,
    model_name,
    augmentation,
    batch_size=32,
    learning_rate=1e-3,
    epochs=40,
    patience=8,
    focal_gamma=None,
):
    tf.keras.backend.clear_session()
    setup_gpu()
    data = load_data()
    output_dir = RESULTS_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "split_indices.npz",
        train_indices=data["train_indices"],
        val_indices=data["val_indices"],
    )
    (output_dir / "classes.json").write_text(
        json.dumps(data["class_names"], indent=2), encoding="utf-8"
    )

    configuration = {
        "run_name": run_name,
        "model": model_name,
        "augmentation": augmentation,
        "loss": "focal" if focal_gamma is not None else "cross_entropy",
        "focal_gamma": focal_gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping_patience": patience,
    }
    (output_dir / "hyperparameters.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    train_ds = make_dataset(
        data["train_x"],
        data["train_y"],
        batch_size,
        training=True,
        augment=augmentation,
    )
    val_ds = make_dataset(data["val_x"], data["val_y"], batch_size)
    test_ds = make_dataset(data["test_x"], data["test_y"], batch_size)

    loss_function = (
        SparseCategoricalFocalLoss(gamma=focal_gamma)
        if focal_gamma is not None
        else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    tf.keras.utils.set_random_seed(SEED)
    model = build_plain_cnn() if model_name == "plain_cnn" else build_lenet5()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=loss_function,
        metrics=make_metrics(),
    )

    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "best.h5"),
            monitor="val_top1",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "history.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top1",
            mode="max",
            patience=patience,
            restore_best_weights=False,
        ),
    ]
    if focal_gamma is not None:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    save_learning_curves(history, output_dir)

    best_model = tf.keras.models.load_model(str(output_dir / "best.h5"), compile=False)
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=loss_function,
        metrics=make_metrics(),
    )
    validation = best_model.evaluate(val_ds, return_dict=True, verbose=1)
    test_metrics = save_test_results(
        best_model,
        test_ds,
        data["test_y"],
        data["class_names"],
        output_dir,
        test_x=data["test_x"],
        focal_gamma=focal_gamma,
    )

    metrics = {
        **configuration,
        "parameters": int(best_model.count_params()),
        "best_epoch": int(np.argmax(history.history["val_top1"]) + 1),
        "val_top1": float(validation["top1"]),
        "val_top3": float(validation["top3"]),
        "val_top5": float(validation["top5"]),
        **test_metrics,
        "split_hash": data["split_hash"],
        "train_images": int(len(data["train_y"])),
        "validation_images": int(len(data["val_y"])),
        "test_images": int(len(data["test_y"])),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    update_comparison()
    print(json.dumps(metrics, indent=2))

