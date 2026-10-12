"""不下載資料或預訓練權重；檢查真實 TF2.11 的 forward/backward 與序列遮罩。"""
import numpy as np
import tensorflow as tf
from .common import RESULTS, save_json


def run(args):
    from .sentiment import build as sentiment
    from .caption import build as caption, masked_loss, decode, make_tokenizer, caption_dataset
    from .vision import build as vision, compile_model, unfreeze
    checks = []
    for kind in ('rnn', 'lstm'):
        tf.keras.backend.clear_session()
        model = sentiment(kind, vocab=32, length=8)
        x = np.random.default_rng(42).integers(1, 32, (2, 8), dtype='int32')
        value = model.train_on_batch(x, np.asarray([0., 1.], dtype='float32'))
        assert np.isfinite(value).all()
        checks.append('sentiment_' + kind + ': forward/backward OK')
    for kind in ('gru', 'lstm'):
        tf.keras.backend.clear_session()
        model = caption(kind, vocab=32, length=8, feature_dim=16)
        features = np.zeros((2, 16), dtype='float32')
        tokens = np.asarray([[1, 3, 4, 2, 0, 0, 0], [1, 5, 2, 0, 0, 0, 0]], dtype='int32')
        target = np.asarray([[3, 4, 2, 0, 0, 0, 0], [5, 2, 0, 0, 0, 0, 0]], dtype='int32')
        values = model.train_on_batch([features, tokens], target)
        assert np.isfinite(values).all()
        zeros = tf.zeros((1, 7), tf.int32)
        assert float(masked_loss(zeros, tf.zeros((1, 7, 32)))) == 0
        # Adding padding must not change a real token's loss.
        one_loss = masked_loss(tf.constant([[3]]), tf.zeros((1, 1, 32)))
        padded_loss = masked_loss(tf.constant([[3, 0, 0]]), tf.zeros((1, 3, 32)))
        np.testing.assert_allclose(one_loss.numpy(), padded_loss.numpy(), rtol=1e-6)
        path = RESULTS / 'smoke' / (kind + '.weights.h5')
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(path))
        model.load_weights(str(path))
        checks.append('caption_' + kind + ': forward/backward, padding, weight roundtrip OK')
    # Real ViT-B16 architecture; freeze backbone to keep CPU memory use manageable.
    for kind in ('vit', 'resnet'):
        tf.keras.backend.clear_session()
        model, backbone = vision(kind, pretrained=False)
        compile_model(model, 1e-3)
        x = np.zeros((1, 224, 224, 3), dtype='float32')
        values = model.train_on_batch(x, np.asarray([0], dtype='int32'))
        assert np.isfinite(values).all()
        unfreeze(backbone, kind)
        assert sum(np.prod(w.shape) for w in backbone.trainable_weights) > 0
        compile_model(model, 1e-5)
        values = model.train_on_batch(x, np.asarray([0], dtype='int32'))
        assert np.isfinite(values).all()
        checks.append(kind + ': warmup and backbone finetune backward OK')
    # Deterministic decoder: start -> word -> end, both greedy and beam search.
    class Tokens:
        word_index = {'startseq': 1, 'endseq': 2}
        index_word = {1: 'startseq', 2: 'endseq', 3: 'word'}
    def fake_infer(features, tokens):
        output = np.full((len(tokens), 7, 4), -100., dtype='float32')
        output[:, 0, 3] = 100
        output[:, 1:, 2] = 100
        return tf.constant(output)
    for width in (1, 3):
        assert decode(fake_infer, np.zeros(16), Tokens(), length=8, beam_width=width) == 'word'
    checks.append('greedy/beam3 EOS and token filtering OK')
    tokenizer = make_tokenizer({'train.jpg': ['a dog runs'], 'test.jpg': ['unseenword']}, ['train.jpg'])
    assert 'unseenword' not in tokenizer.word_index
    dataset = caption_dataset(['train.jpg'], {'train.jpg': ['a dog runs']}, tokenizer,
                              np.zeros((1, 1280), dtype='float32'), 1, args.seed, False)
    (feature, prefix), target = next(iter(dataset))
    assert int(prefix[0, 0]) == tokenizer.word_index['startseq']
    assert int(target[0, 3]) == tokenizer.word_index['endseq']
    checks.append('caption train-only vocabulary and teacher-forcing alignment OK')
    save_json(RESULTS / 'smoke' / 'checks.json', {'tensorflow': tf.__version__, 'checks': checks,
        'scope': 'synthetic samples, no pretrained weights or real dataset training'})
    print('\n'.join(checks))
