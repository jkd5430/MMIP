"""只在 train fit 詞彙表；Flickr8k 以影像切分，五筆描述不跨 split。"""
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .common import DATA, RESULTS, save_json


def load_imdb(args):
    import tensorflow_datasets as tfds
    ds, info = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True,
                         with_info=True, data_dir=str(DATA / 'tfds'), shuffle_files=False)
    def materialize(source):
        pairs = list(tfds.as_numpy(source))
        return np.asarray([x.decode('utf-8') for x, _ in pairs], dtype=object), np.asarray([y for _, y in pairs])
    texts, labels = materialize(ds[0])
    test_x, test_y = materialize(ds[1])
    train_i, val_i = train_test_split(np.arange(len(labels)), test_size=0.2,
                                    stratify=labels, random_state=args.seed)
    splits = {'train': (texts[train_i], labels[train_i]),
              'validation': (texts[val_i], labels[val_i]), 'test': (test_x, test_y)}
    if args.limit:
        splits = {key: (x[:args.limit], y[:args.limit]) for key, (x, y) in splits.items()}
    return splits


def load_beans(args):
    import tensorflow_datasets as tfds
    split_names = ('train', 'validation', 'test')
    # Reuse an already prepared TFDS cache without calling its obsolete GCS URLs.
    builder = tfds.builder('beans', data_dir=str(DATA / 'tfds'))
    if all(name in builder.info.splits for name in split_names):
        datasets = [builder.as_dataset(split=name, as_supervised=True, shuffle_files=False)
                    for name in split_names]
        classes = builder.info.features['label'].names
    else:
        print('Beans: using the author-hosted Hugging Face files (old TFDS URL returns HTTP 403).')
        datasets, classes = load_beans_hf()
    if args.limit:
        datasets = [ds.take(args.limit) for ds in datasets]
    return dict(zip(split_names, datasets)), classes


def load_beans_hf():
    """Read the author's Parquet files directly; no Hugging Face datasets SDK needed."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError('Beans 替代來源需要 pyarrow：python -m pip install pyarrow==12.0.1') from error
    import requests

    revision = '27aa014ce09b193e1a6f58112d4a66e0eddb69c5'
    repository = 'https://huggingface.co/datasets/AI-Lab-Makerere/beans'
    classes = ['angular_leaf_spot', 'bean_rust', 'healthy']
    counts = {'train': 1034, 'validation': 133, 'test': 128}
    root = DATA / 'beans_hf'
    root.mkdir(parents=True, exist_ok=True)
    datasets = []
    for split, count in counts.items():
        path = root / (split + '-00000-of-00001.parquet')
        if not path.exists():
            url = repository + '/resolve/' + revision + '/data/' + path.name
            partial = path.with_suffix('.parquet.part')
            print('Downloading Beans ' + split + ' from author repository', flush=True)
            with requests.get(url, stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                with partial.open('wb') as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if pq.ParquetFile(partial).metadata.num_rows != count:
                raise ValueError('Beans ' + split + ': downloaded row count does not match official split')
            partial.replace(path)
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != count:
            raise ValueError('Beans cache row count mismatch: ' + str(path))
        schema_metadata = parquet.schema_arrow.metadata or {}
        if b'huggingface' in schema_metadata:
            metadata = json.loads(schema_metadata[b'huggingface'])
            label_names = metadata.get('info', {}).get('features', {}).get('labels', {}).get('names')
            if label_names is not None and label_names != classes:
                raise ValueError('Beans label mapping differs from the expected three classes')

        def rows(filename=path):
            for batch in pq.ParquetFile(filename).iter_batches(batch_size=64, columns=['image', 'labels']):
                values = batch.to_pydict()
                for image, label in zip(values['image'], values['labels']):
                    if not image or not image.get('bytes') or not 0 <= label < len(classes):
                        raise ValueError('Invalid Beans image or label')
                    yield image['bytes'], np.int64(label)

        ds = tf.data.Dataset.from_generator(rows, output_signature=(
            tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.int64)))
        ds = ds.apply(tf.data.experimental.assert_cardinality(count))
        ds = ds.map(lambda encoded, label: (tf.ensure_shape(
            tf.io.decode_image(encoded, channels=3, expand_animations=False), (None, None, 3)), label),
            num_parallel_calls=tf.data.AUTOTUNE)
        datasets.append(ds)
    save_json(root / 'source.json', {'repository': repository, 'revision': revision,
              'split_samples': counts, 'classes': classes, 'splits': 'official, unchanged'})
    return datasets, classes


def download_archive(url, root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / url.rsplit('/', 1)[-1]
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        print('Downloading:', url, flush=True)
        urlretrieve(url, partial)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (root / item.filename).resolve()
            if root.resolve() not in target.parents and target != root.resolve():
                raise ValueError('Unsafe ZIP member: ' + item.filename)
        z.extractall(root)


def clean_caption(text):
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))


def load_flickr(args):
    root = DATA / 'flickr8k'
    root.mkdir(parents=True, exist_ok=True)
    tokens = list(root.rglob('Flickr8k.token.txt'))
    csv_files = list(root.rglob('captions.txt'))
    if not tokens and not csv_files:
        download_archive('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', root)
        tokens = list(root.rglob('Flickr8k.token.txt'))
    images = {p.name: p for p in root.rglob('*.jpg') if '__MACOSX' not in p.parts}
    if not images:
        download_archive('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', root)
        images = {p.name: p for p in root.rglob('*.jpg') if '__MACOSX' not in p.parts}
    captions = defaultdict(list)
    if tokens:
        for line in tokens[0].read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            image_id, caption = line.split('\t', 1)
            captions[image_id.split('#')[0]].append(clean_caption(caption))
    else:
        with csv_files[0].open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                captions[row['image']].append(clean_caption(row['caption']))
    splits = {}
    for name, file in [('train', 'Flickr_8k.trainImages.txt'),
                       ('validation', 'Flickr_8k.devImages.txt'), ('test', 'Flickr_8k.testImages.txt')]:
        found = list(root.rglob(file))
        if found:
            splits[name] = [s.strip() for s in found[0].read_text().splitlines() if s.strip()]
    # The upstream text ZIP has five captions for this orphan id, but neither
    # the image ZIP nor any official split contains it. Do not guess a filename
    # or silently drop other missing images, especially ones required by a split.
    known_orphans = {'2258277193_586949ec62.jpg.1'}
    required = {name for names in splits.values() for name in names}
    missing = set(captions) - set(images)
    excluded = []
    for name in sorted((missing & known_orphans) - required):
        excluded.append({'image': name, 'caption_count': len(captions[name]),
                         'reason': 'upstream orphan annotation; no image and unused by official splits'})
        del captions[name]
        print('Flickr8k: excluding known orphan annotation ' + name + ' (official splits unchanged).')
    save_json(root / 'annotation_issues.json', {'excluded_annotations': excluded})
    missing = set(captions) - set(images)
    if missing:
        raise FileNotFoundError(f'Flickr8k 缺少 {len(missing)} 張影像，例如 {sorted(missing)[:3]}')
    if not splits:
        names = np.asarray(sorted(captions))
        names = np.random.default_rng(args.seed).permutation(names).tolist()
        a, b = int(len(names) * .75), int(len(names) * .875)
        splits = {'train': names[:a], 'validation': names[a:b], 'test': names[b:]}
        strategy = 'seeded image-level 75/12.5/12.5 split (captions.txt input)'
    elif len(splits) != 3:
        raise ValueError('Flickr8k 官方 split 檔案不完整，須同時提供 train/dev/test。')
    else:
        strategy = 'official image-level train/dev/test split'
    sets = [set(splits[k]) for k in ('train', 'validation', 'test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Flickr8k splits overlap')
    for names in splits.values():
        if len(set(names)) != len(names):
            raise ValueError('Duplicate image in split')
        if any(n not in captions or n not in images for n in names):
            raise ValueError('Flickr8k split contains an image without captions or file')
    save_json(root / 'split_manifest.json', {'strategy': strategy, 'seed': args.seed,
              'splits': splits, 'excluded_annotations': excluded})
    if args.limit:
        splits = {key: names[:args.limit] for key, names in splits.items()}
    return images, dict(captions), splits


def prepare(args):
    out = RESULTS / 'q1_data'
    out.mkdir(parents=True, exist_ok=True)
    imdb = load_imdb(args)
    beans, classes = load_beans(args)
    images, captions, splits = load_flickr(args)
    summary = {
        'imdb': {'split_samples': {k: len(x) for k, (x, _) in imdb.items()},
                 'classes': ['negative', 'positive'], 'validation': 'stratified 20% of official train',
                 'test': 'official test', 'train_examples': imdb['train'][0][:3].tolist()},
        'beans': {'split_samples': {k: int(tf.data.experimental.cardinality(ds)) for k, ds in beans.items()},
                  'classes': classes, 'splits': 'official train/validation/test'},
        'flickr8k': {'split_images': {k: len(v) for k, v in splits.items()},
                     'split_caption_pairs': {k: sum(len(captions[n]) for n in v) for k, v in splits.items()},
                     'examples': {n: captions[n] for n in splits['train'][:3]}},
        'debug_subset': args.limit is not None,
    }
    save_json(out / 'dataset_summary.json', summary)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (image, label) in zip(axes[0], beans['train'].take(3)):
        axis.imshow(image.numpy())
        axis.set_title(classes[int(label)])
        axis.axis('off')
    for axis, name in zip(axes[1], splits['train'][:3]):
        axis.imshow(plt.imread(images[name]))
        axis.set_title(captions[name][0], fontsize=8, wrap=True)
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'dataset_examples.png', dpi=150)
    plt.close(fig)
    print('Dataset summary:', out / 'dataset_summary.json')
