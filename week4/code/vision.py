import time
import numpy as np
import tensorflow as tf
from .common import directory, fit, save_json, model_summary, classification
from .datasets import load_beans

SIZE = 224


def build(kind, pretrained=True):
    if kind == 'vit':
        from vit_keras import vit
        backbone = vit.vit_b16(image_size=SIZE, pretrained=pretrained, include_top=False,
                               pretrained_top=False, weights='imagenet21k+imagenet2012')
    else:
        backbone = tf.keras.applications.ResNet50(include_top=False, pooling='avg',
                        weights='imagenet' if pretrained else None, input_shape=(SIZE, SIZE, 3))
    backbone.trainable = False
    images = tf.keras.Input((SIZE, SIZE, 3), name='images_0_to_255')
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(.05),
        tf.keras.layers.RandomZoom(.1),
    ], name='train_augmentation')
    x = augmentation(images)
    if kind == 'vit':
        x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1.)(x)
    else:
        x = tf.keras.applications.resnet50.preprocess_input(x)
    # Keeps ResNet BN moving statistics fixed; unfrozen weights still receive gradients.
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='beans_classifier')(x)
    return tf.keras.Model(images, outputs, name=kind), backbone


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def unfreeze(backbone, kind):
    backbone.trainable = True
    # ViT: last two transformer blocks and final normalization.
    # ResNet: last residual stage; BN remains frozen on the small Beans dataset.
    for layer in backbone.layers:
        if kind == 'vit':
            layer.trainable = layer.name.startswith(('Transformer/encoderblock_10',
                                'Transformer/encoderblock_11', 'Transformer/encoder_norm'))
        else:
            layer.trainable = layer.name.startswith('conv5_') and not isinstance(layer, tf.keras.layers.BatchNormalization)
    if not any(layer.trainable_weights for layer in backbone.layers):
        raise RuntimeError('未解凍任何 backbone 權重；請確認 vit-keras 版本為 0.1.2。')


def run(args, kind):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    out = directory('q3_' + kind, args)
    raw, classes = load_beans(args)
    datasets = {}
    for key, ds in raw.items():
        ds = ds.map(lambda image, label: (tf.image.resize(tf.cast(image, tf.float32), (SIZE, SIZE)), label),
                     num_parallel_calls=tf.data.AUTOTUNE).cache()
        if key == 'train':
            ds = ds.shuffle(1034, seed=args.seed)
        datasets[key] = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    model, backbone = build(kind)
    model_summary(model, out)
    compile_model(model, 1e-3)
    head_time, head_loss = fit(model, datasets['train'], datasets['validation'], out,
                               args.warmup_epochs, 'warmup')
    unfreeze(backbone, kind)
    save_json(out / 'unfrozen_layers.json', [layer.name for layer in backbone.layers if layer.trainable_weights])
    compile_model(model, 1e-5)
    fine_time, fine_loss = fit(model, datasets['train'], datasets['validation'], out, args.epochs, 'finetune')
    # Choose using validation only; test never participates in checkpoint selection.
    selected = 'finetune' if fine_loss <= head_loss else 'warmup'
    model.load_weights(str(out / (selected + '.weights.h5')))
    model.save_weights(str(out / 'best.weights.h5'))
    labels = np.concatenate([y.numpy() for _, y in datasets['test']])
    model(next(iter(datasets['test']))[0][:1], training=False).numpy()
    start = time.perf_counter()
    probabilities = model.predict(datasets['test'], verbose=0)
    elapsed = time.perf_counter() - start
    metrics = classification(out, labels, probabilities, classes,
                [{'test_index': i} for i in range(len(labels))], head_time + fine_time)
    metrics.update(parameters=model.count_params(), selected_phase=selected,
                   inference_ms_per_image=elapsed * 1000 / len(labels),
                   pretrained_backbone=True, finetuned_backbone=True,
                   warmup_val_loss=head_loss, finetune_val_loss=fine_loss)
    save_json(out / 'metrics.json', metrics)
    import matplotlib.pyplot as plt
    samples = list(raw['test'].take(9))
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for i, axis in enumerate(axes.flat):
        axis.axis('off')
        if i < len(samples):
            axis.imshow(samples[i][0].numpy())
            axis.set_title(f'True: {classes[int(labels[i])]}\nPred: {classes[int(probabilities[i].argmax())]}', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'test_predictions.png', dpi=150)
    plt.close(fig)
