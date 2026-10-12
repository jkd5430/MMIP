"""Flickr8k: ImageNet MobileNetV2 encoder + GRU/LSTM decoder, greedy/beam inference."""
import json
import time
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from .common import DATA, RESULTS, directory, fit, model_summary, save_json, save_csv
from .datasets import clean_caption, load_flickr

MAX_LENGTH = 40
VOCAB = 5000
FEATURE_DIM = 1280


def masked_loss(y_true, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(y_true, 0), loss.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(loss * mask), tf.reduce_sum(mask))


def masked_accuracy(y_true, logits):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    correct = tf.cast(tf.equal(tf.cast(y_true, tf.int64), tf.argmax(logits, axis=-1)), tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(correct * mask), tf.reduce_sum(mask))


def build(kind, vocab, length=MAX_LENGTH, feature_dim=FEATURE_DIM):
    feature = tf.keras.Input((feature_dim,), name='image_feature')
    tokens = tf.keras.Input((length - 1,), dtype='int32', name='caption_prefix')
    x = tf.keras.layers.Embedding(vocab, 256, mask_zero=True)(tokens)
    initial_h = tf.keras.layers.Dense(256, activation='tanh', name='image_to_h')(feature)
    if kind == 'lstm':
        initial_c = tf.keras.layers.Dense(256, activation='tanh', name='image_to_c')(feature)
        x = tf.keras.layers.LSTM(256, return_sequences=True)(x, initial_state=[initial_h, initial_c])
    else:
        x = tf.keras.layers.GRU(256, return_sequences=True)(x, initial_state=initial_h)
    x = tf.keras.layers.Dropout(.3)(x)
    logits = tf.keras.layers.Dense(vocab)(x)
    # Padding is handled explicitly by the loss; remove the Keras output mask.
    logits = RemoveMask()(logits)
    model = tf.keras.Model([feature, tokens], logits, name='caption_' + kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=masked_loss,
                   metrics=[tf.keras.metrics.MeanMetricWrapper(masked_accuracy, name='accuracy')])
    return model


class RemoveMask(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.identity(inputs)

    def compute_mask(self, inputs, mask=None):
        return None


def encoder_model(pretrained=True):
    return tf.keras.applications.MobileNetV2(weights='imagenet' if pretrained else None,
               include_top=False, pooling='avg', input_shape=(224, 224, 3))


def read_image(path):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(tf.cast(image, tf.float32), (224, 224))
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def extract_features(images, names, encoder, batch_size):
    cache = DATA / 'flickr8k' / 'features_mobilenetv2_imagenet_224_avg'
    cache.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (cache / (n + '.npy')).exists()]
    if missing:
        ds = tf.data.Dataset.from_tensor_slices([str(images[n]) for n in missing])
        ds = ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        cursor = 0
        for batch in ds:
            values = encoder(batch, training=False).numpy()
            for value in values:
                np.save(cache / (missing[cursor] + '.npy'), value.astype('float32'))
                cursor += 1
            print(f'CNN features {cursor}/{len(missing)}', end='\r', flush=True)
        print()
    values = np.stack([np.load(cache / (n + '.npy'), allow_pickle=False) for n in names])
    if values.shape != (len(names), FEATURE_DIM) or not np.isfinite(values).all():
        raise ValueError('CNN feature cache shape/value invalid; remove the affected .npy files and rerun')
    return values


def make_tokenizer(captions, train_names):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB, oov_token='<unk>', filters='')
    tokenizer.fit_on_texts(['startseq ' + captions[n][i] + ' endseq'
                           for n in train_names for i in range(len(captions[n]))])
    if max(tokenizer.word_index['startseq'], tokenizer.word_index['endseq']) >= VOCAB:
        raise ValueError('Start/end token outside vocabulary')
    return tokenizer


def caption_dataset(names, captions, tokenizer, features, batch_size, seed, training):
    ids, sequences = [], []
    for i, name in enumerate(names):
        for text in captions[name]:
            words = tokenizer.texts_to_sequences([text])[0][:MAX_LENGTH - 2]
            sequences.append([tokenizer.word_index['startseq']] + words + [tokenizer.word_index['endseq']])
            ids.append(i)
    tokens = tf.keras.utils.pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    feature_tensor = tf.constant(features)
    ds = tf.data.Dataset.from_tensor_slices((np.asarray(ids, dtype='int32'), tokens))
    if training:
        ds = ds.shuffle(len(ids), seed=seed)
    ds = ds.batch(batch_size).map(
        lambda indices, sequence: ((tf.gather(feature_tensor, indices), sequence[:, :-1]), sequence[:, 1:]),
        num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def decode(infer, feature, tokenizer, length=MAX_LENGTH, beam_width=1):
    """Length-normalized beam search; width=1 is greedy. Never emits pad/start."""
    start = tokenizer.word_index['startseq']
    end = tokenizer.word_index['endseq']
    beams = [([start], 0.0)]
    def score(item):
        tokens, logp = item
        return logp / (((5.0 + len(tokens) - 1) / 6.0) ** .7)
    for _ in range(length - 1):
        active = [(tokens, logp) for tokens, logp in beams if tokens[-1] != end]
        if not active:
            break
        prefixes = tf.keras.utils.pad_sequences([t for t, _ in active], maxlen=length - 1, padding='post')
        features = np.repeat(np.asarray(feature)[None, :], len(active), axis=0)
        logits = infer(tf.convert_to_tensor(features, tf.float32), tf.convert_to_tensor(prefixes, tf.int32)).numpy()
        candidates = [b for b in beams if b[0][-1] == end]
        for j, (tokens, logp) in enumerate(active):
            log_probs = tf.nn.log_softmax(logits[j, len(tokens) - 1]).numpy()
            log_probs[0] = -np.inf
            log_probs[start] = -np.inf
            best = np.argsort(log_probs)[-beam_width:][::-1]
            candidates.extend((tokens + [int(token)], logp + float(log_probs[token])) for token in best)
        beams = sorted(candidates, key=score, reverse=True)[:beam_width]
    tokens = max(beams, key=score)[0]
    return ' '.join(tokenizer.index_word.get(i, '<unk>') for i in tokens if i not in (0, start, end))


def require_bleu():
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError as error:
        raise RuntimeError('BLEU 需要 nltk；請在目前 Python 環境執行：'
                           'python -m pip install nltk==3.8.1。'
                           '若已有 Q4 權重，可用 --evaluate-only 接續評估。') from error
    return corpus_bleu, SmoothingFunction


def bleu_scores(rows, method):
    corpus_bleu, SmoothingFunction = require_bleu()
    references = [[clean_caption(s).split() for s in row['references']] for row in rows]
    hypotheses = [row[method].split() for row in rows]
    values = {}
    for n in range(1, 5):
        values[f'bleu_{n}'] = float(corpus_bleu(references, hypotheses,
                weights=tuple([1.0 / n] * n), smoothing_function=SmoothingFunction().method1))
    return values


def run(args, kind):
    require_bleu()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    if getattr(args, 'evaluate_only', False):
        out = RESULTS / ('q4_' + kind)
        required_files = ('best.weights.h5', 'tokenizer.json', 'preprocessing.json', 'hyperparameters.json')
        missing = [name for name in required_files if not (out / name).exists()]
        if missing:
            raise FileNotFoundError('Q4 評估所需檔案不存在：' + str(out) + ' / ' + ', '.join(missing))
        saved = json.loads((out / 'hyperparameters.json').read_text(encoding='utf-8'))
        config = json.loads((out / 'preprocessing.json').read_text(encoding='utf-8'))
        if config['kind'] != kind:
            raise ValueError('Saved caption model kind differs from the selected task')
        # Reuse training-time split settings and tokenizer. Do not overwrite them
        # with evaluation CLI defaults or refit a vocabulary on new data.
        data_args = SimpleNamespace(seed=saved['seed'], limit=saved.get('limit'))
        images, captions, splits = load_flickr(data_args)
        actual_counts = {name: len(values) for name, values in splits.items()}
        if actual_counts != config['split_images']:
            raise ValueError('Flickr8k split counts differ from the saved training configuration')
        model, tokenizer = load_trained(out)
        train_seconds = None
        for filename in ('training_state.json', 'metrics.json'):
            path = out / filename
            if path.exists():
                train_seconds = json.loads(path.read_text(encoding='utf-8')).get('train_seconds')
                if train_seconds is not None:
                    break
        save_json(out / 'evaluation_config.json', {**vars(args),
                  'training_seed': data_args.seed, 'training_limit': data_args.limit})
        print('Evaluation only: using saved ' + kind.upper() + ' weights and tokenizer; training skipped.')
        encoder = encoder_model()
    else:
        out = directory('q4_' + kind, args)
        images, captions, splits = load_flickr(args)
        tokenizer = make_tokenizer(captions, splits['train'])
        (out / 'tokenizer.json').write_text(tokenizer.to_json(), encoding='utf-8')
        vocab_size = min(VOCAB, len(tokenizer.word_index) + 1)
        save_json(out / 'preprocessing.json', {'vocab_size': vocab_size, 'max_length': MAX_LENGTH,
                  'feature_dim': FEATURE_DIM, 'kind': kind, 'encoder': 'ImageNet MobileNetV2 frozen',
                  'train_vocabulary_only': True, 'split_images': {k: len(v) for k, v in splits.items()},
                  'beam_width': 3, 'beam_length_penalty': .7})
        save_json(out / 'training_split.json', splits)
        encoder = encoder_model()
        train_features = extract_features(images, splits['train'], encoder, args.batch_size)
        val_features = extract_features(images, splits['validation'], encoder, args.batch_size)
        train = caption_dataset(splits['train'], captions, tokenizer, train_features, args.batch_size, args.seed, True)
        val = caption_dataset(splits['validation'], captions, tokenizer, val_features, args.batch_size, args.seed, False)
        model = build(kind, vocab_size)
        model_summary(model, out)
        train_seconds, _ = fit(model, train, val, out, args.epochs)
        model.save_weights(str(out / 'best.weights.h5'))
        save_json(out / 'training_state.json', {'train_seconds': train_seconds, 'training_completed': True})

    split_snapshot = out / 'training_split.json'
    if split_snapshot.exists() and json.loads(split_snapshot.read_text(encoding='utf-8')) != splits:
        raise ValueError('Flickr8k split membership/order differs from the saved training split')

    @tf.function(input_signature=[tf.TensorSpec((None, FEATURE_DIM), tf.float32),
                                  tf.TensorSpec((None, MAX_LENGTH - 1), tf.int32)])
    def infer(features, tokens):
        return model([features, tokens], training=False)

    selected = splits['test'][:args.eval_images] if args.eval_images else splits['test']
    # Warm up encoder and both batch sizes before measuring inference.
    warm_feature = encoder(read_image(str(images[selected[0]]))[None], training=False).numpy()[0]
    for width in (1, 3):
        decode(infer, warm_feature, tokenizer, beam_width=width)
    rows = []
    for i, name in enumerate(selected):
        start_time = time.perf_counter()
        feature = encoder(read_image(str(images[name]))[None], training=False).numpy()[0]
        encoder_ms = (time.perf_counter() - start_time) * 1000
        row = {'image': name, 'image_path': str(images[name].relative_to(DATA.parent)),
                'references': captions[name], 'encoder_ms': encoder_ms}
        for method, width in [('greedy', 1), ('beam3', 3)]:
            start_time = time.perf_counter()
            row[method] = decode(infer, feature, tokenizer, beam_width=width)
            row[method + '_decoder_ms'] = (time.perf_counter() - start_time) * 1000
            row[method + '_total_ms'] = encoder_ms + row[method + '_decoder_ms']
        rows.append(row)
        if (i + 1) % 25 == 0:
            save_json(out / 'predictions.partial.json', rows)
            print(f'Caption evaluation {i + 1}/{len(selected)}', flush=True)
    # Preserve generated captions before computing external-library metrics.
    save_json(out / 'predictions.json', rows)
    metrics = {'train_seconds': train_seconds, 'parameters': model.count_params(),
                'evaluated_images': len(rows), 'test_split_images': len(splits['test']),
                'test_subset': len(rows) != len(splits['test']),
                'bleu_scale': '0..1, corpus BLEU, all reference captions, smoothing method1',
                'timing': 'warm batch=1 encoder (JPEG read/resize/CNN) + decoder; encoder shared between methods'}
    comparison = []
    for method in ('greedy', 'beam3'):
        values = bleu_scores(rows, method)
        values.update(mean_decoder_ms=float(np.mean([r[method + '_decoder_ms'] for r in rows])),
                       mean_total_ms=float(np.mean([r[method + '_total_ms'] for r in rows])),
                       median_total_ms=float(np.median([r[method + '_total_ms'] for r in rows])))
        metrics[method] = values
        comparison.append({'method': kind + '_' + method, **values})
    save_json(out / 'metrics.json', metrics)
    save_csv(out / 'comparison.csv', comparison)
    import matplotlib.pyplot as plt
    import textwrap
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    for i, axis in enumerate(axes.flat):
        axis.axis('off')
        if i < len(rows):
            row = rows[i]
            axis.imshow(plt.imread(images[row['image']]))
            title = 'Reference: ' + row['references'][0] + '\nGreedy: ' + row['greedy'] + '\nBeam3: ' + row['beam3']
            axis.set_title('\n'.join(textwrap.fill(line, 65) for line in title.split('\n')), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'caption_examples.png', dpi=150)
    plt.close(fig)


def load_trained(out):
    """重建模型供後續影像推論，不需要重新訓練。"""
    config = json.loads((out / 'preprocessing.json').read_text(encoding='utf-8'))
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json((out / 'tokenizer.json').read_text(encoding='utf-8'))
    model = build(config['kind'], config['vocab_size'], config['max_length'], config['feature_dim'])
    model.load_weights(str(out / 'best.weights.h5'))
    return model, tokenizer
