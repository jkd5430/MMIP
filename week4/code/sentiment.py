import time
import numpy as np
import tensorflow as tf
from .common import directory, fit, save_json, model_summary, classification
from .datasets import load_imdb

VOCAB = 20000
LENGTH = 256


def build(kind, vocab=VOCAB, length=LENGTH):
    tokens = tf.keras.Input((length,), dtype='int32', name='tokens')
    x = tf.keras.layers.Embedding(vocab, 128, mask_zero=True)(tokens)
    recurrent = tf.keras.layers.SimpleRNN if kind == 'rnn' else tf.keras.layers.LSTM
    x = recurrent(64, dropout=0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(tokens, output, name=kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run(args, kind):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    out = directory('q2_' + kind, args)
    splits = load_imdb(args)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB, oov_token='<unk>')
    tokenizer.fit_on_texts(splits['train'][0])
    (out / 'tokenizer.json').write_text(tokenizer.to_json(), encoding='utf-8')
    save_json(out / 'preprocessing.json', {'max_length': LENGTH, 'vocab_limit': VOCAB,
              'padding': 'post', 'truncating': 'post', 'fit_vocabulary_on': 'train only',
              'split_samples': {k: len(x) for k, (x, y) in splits.items()}})
    arrays = {}
    datasets = {}
    for key, (texts, labels) in splits.items():
        x = tf.keras.utils.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=LENGTH,
                                         padding='post', truncating='post')
        arrays[key] = x
        ds = tf.data.Dataset.from_tensor_slices((x, labels.astype('float32')))
        if key == 'train':
            ds = ds.shuffle(len(x), seed=args.seed)
        datasets[key] = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    model = build(kind)
    model_summary(model, out)
    seconds, _ = fit(model, datasets['train'], datasets['validation'], out, args.epochs)
    model.save(str(out / 'model.h5'))
    # Warm up, then measure batch throughput. .numpy() synchronizes GPU work.
    model(arrays['test'][:1], training=False).numpy()
    start = time.perf_counter()
    probabilities = model.predict(datasets['test'], verbose=0)
    inference = time.perf_counter() - start
    texts, labels = splits['test']
    metrics = classification(out, labels, probabilities, ['negative', 'positive'],
                              [{'review': str(text)} for text in texts], seconds)
    metrics.update(parameters=model.count_params(), inference_ms_per_review=inference * 1000 / len(labels))
    save_json(out / 'metrics.json', metrics)
