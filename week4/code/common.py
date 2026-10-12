import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, auc)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def setup(seed):
    if not tf.__version__.startswith('2.11.'):
        raise RuntimeError('本作業要求 TensorFlow 2.11.x，目前為 ' + tf.__version__)
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    print('TensorFlow', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
    for folder in (DATA, RESULTS, ROOT / 'ai'):
        folder.mkdir(parents=True, exist_ok=True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def directory(task, args):
    out = RESULTS / task
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'hyperparameters.json', {**vars(args), 'experiment': task,
              'tensorflow': tf.__version__, 'debug_subset': args.limit is not None})
    return out


def fit(model, train, val, out, epochs, phase='train'):
    checkpoint = out / (phase + '.weights.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(out / (phase + '_epochs.csv'))),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)
    model.load_weights(str(checkpoint))
    save_json(out / (phase + '_history.json'), history.history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ('loss', 'accuracy')):
        for key in (metric, 'val_' + metric):
            if key in history.history:
                axis.plot(np.arange(1, len(history.history[key]) + 1), history.history[key], marker='o', label=key)
        axis.set(xlabel='Epoch', title=metric)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(out / (phase + '_learning_curves.png'), dpi=150)
    plt.close(fig)
    return time.perf_counter() - start, min(history.history['val_loss'])


def model_summary(model, out):
    with (out / 'model_summary.txt').open('w', encoding='utf-8') as f:
        model.summary(print_fn=lambda line: f.write(line + '\n'))


def classification(out, labels, probabilities, class_names, rows, train_seconds):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    binary = len(class_names) == 2
    if binary:
        p = probabilities.reshape(-1)
        probabilities = np.column_stack((1 - p, p))
    predictions = probabilities.argmax(axis=1)
    target = np.eye(len(class_names))[labels]
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0),
        'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
        'macro_auc': None, 'train_seconds': train_seconds, 'test_samples': len(labels),
        'parameters': None,
    }
    if len(np.unique(labels)) == len(class_names):
        metrics['macro_auc'] = float(roc_auc_score(labels, probabilities[:, 1]) if binary else
                                     roc_auc_score(labels, probabilities, average='macro', multi_class='ovr'))
    save_json(out / 'classification_report.json', classification_report(
        labels, predictions, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0))
    for i, row in enumerate(rows):
        row.update(true_class=class_names[labels[i]], predicted_class=class_names[predictions[i]],
                   probabilities=probabilities[i].tolist())
    save_json(out / 'predictions.json', rows)
    save_json(out / 'classes.json', class_names)
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(cm, cmap='Blues')
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
             xticklabels=class_names, yticklabels=class_names, xlabel='Predicted', ylabel='True')
    plt.setp(axis.get_xticklabels(), rotation=25, ha='right')
    for (i, j), count in np.ndenumerate(cm):
        axis.text(j, i, str(count), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5))
    for c, name in enumerate(class_names):
        if len(np.unique(target[:, c])) != 2:
            continue
        fpr, tpr, _ = roc_curve(target[:, c], probabilities[:, c])
        axis.plot(fpr, tpr, label=f'{name}: AUC={auc(fpr, tpr):.4f}')
    axis.plot([0, 1], [0, 1], 'k--')
    axis.set(xlabel='False positive rate', ylabel='True positive rate', title='One-vs-rest ROC')
    if len(axis.lines) > 1:
        axis.legend()
    fig.tight_layout()
    fig.savefig(out / 'roc_curves.png', dpi=150)
    plt.close(fig)
    return metrics
