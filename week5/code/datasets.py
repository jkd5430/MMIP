"""Reuse local Week 4 data; preserve official image-level splits."""
import hashlib
import io
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .common import DATA, ROOT, save_json

SPLITS = ('train', 'validation', 'test')
BEAN_CLASSES = ['angular_leaf_spot', 'bean_rust', 'healthy']


def week4_candidates(args):
    options = [args.week4_dir] if args.week4_dir else []
    return [Path(p) for p in options + [ROOT.parent / 'week4', Path.home() / 'Downloads' / 'MMIP' / 'week4']]


def find_data(args, explicit, folder):
    if explicit:
        return Path(explicit).resolve()
    for project in week4_candidates(args):
        candidate = project / 'data' / folder
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return DATA / folder


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    print('Downloading:', url, flush=True)
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    partial.replace(target)


def beans(args):
    import pyarrow.parquet as pq
    source = find_data(args, args.beans_dir, 'beans_hf')
    result = {}
    expected = dict(zip(SPLITS, [1034, 133, 128]))
    for split in SPLITS:
        filename = split + '-00000-of-00001.parquet'
        path = source / filename
        if not path.exists():
            # Put downloads in this project; never modify a reused Week 4 directory.
            path = DATA / 'beans_hf' / filename
            if not path.exists():
                download('https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/'
                         '27aa014ce09b193e1a6f58112d4a66e0eddb69c5/data/' + filename, path)
        table = pq.read_table(path)
        if table.num_rows != expected[split]:
            raise ValueError('Unexpected official Beans row count: ' + split)
        rows = table.to_pylist()
        if args.limit:
            # Same deterministic subset for CNN/ViT; keep all available classes represented.
            ids = np.random.default_rng(args.seed).permutation(len(rows))[:args.limit]
            rows = [rows[i] for i in ids]
        items = []
        for row in rows:
            content = row['image']['bytes']
            label = int(row['labels'])
            if label not in range(3):
                raise ValueError('Invalid Beans label')
            key = hashlib.sha256(content).hexdigest()
            target = DATA / 'beans_images' / split / (key + '.jpg')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(io.BytesIO(content)) as image:
                    image.convert('RGB').save(target, quality=95)
            items.append({'id': key, 'path': str(target), 'label': label})
        result[split] = items
    sets = [set(r['id'] for r in result[s]) for s in SPLITS]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Beans image content overlaps between splits')
    save_json(args.output_dir / 'beans_manifest.json', {'source': str(source), 'classes': BEAN_CLASSES,
              'official_counts': expected, 'limit': args.limit, 'splits': result})
    return result


def clean(text):
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))


def extract_archive(url, folder):
    archive = DATA / 'downloads' / url.rsplit('/', 1)[-1]
    if not archive.exists():
        download(url, archive)
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (folder / member.filename).resolve()
            if folder.resolve() not in target.parents and target != folder.resolve():
                raise ValueError('Unsafe archive member')
        stream.extractall(folder)


def flickr(args):
    folder = find_data(args, args.flickr_dir, 'flickr8k')
    tokens = list(folder.rglob('Flickr8k.token.txt')) if folder.exists() else []
    if not tokens:
        folder = DATA / 'flickr8k'
        extract_archive('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', folder)
        tokens = list(folder.rglob('Flickr8k.token.txt'))
    images = {p.name: str(p.resolve()) for p in folder.rglob('*.jpg') if '__MACOSX' not in p.parts}
    if not images:
        # Text might have been reused from an incomplete source; keep new files in week5.
        destination = DATA / 'flickr8k'
        images = {p.name: str(p.resolve()) for p in destination.rglob('*.jpg') if '__MACOSX' not in p.parts}
        if not images:
            extract_archive('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', destination)
        images = {p.name: str(p.resolve()) for p in destination.rglob('*.jpg') if '__MACOSX' not in p.parts}
    captions = defaultdict(list)
    for line in tokens[0].read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        key, caption = line.split('\t', 1)
        captions[key.split('#')[0]].append(clean(caption))
    split_files = ['Flickr_8k.trainImages.txt', 'Flickr_8k.devImages.txt', 'Flickr_8k.testImages.txt']
    splits = {}
    for split, filename in zip(SPLITS, split_files):
        found = list(folder.rglob(filename))
        if not found:
            raise FileNotFoundError('Missing official split: ' + filename)
        names = [s.strip() for s in found[0].read_text().splitlines() if s.strip()]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate Flickr image in split')
        if any(n not in images or len(captions[n]) != 5 for n in names):
            raise ValueError('Flickr split requires an image and 5 captions per image')
        splits[split] = names
    sets = [set(splits[s]) for s in SPLITS]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Flickr image-level split overlap')
    counts = {k: len(v) for k, v in splits.items()}
    if args.limit:
        splits = {k: sorted(v)[:args.limit] for k, v in splits.items()}
    result = {s: [{'id': n, 'path': images[n], 'captions': captions[n]} for n in splits[s]] for s in SPLITS}
    save_json(args.output_dir / 'flickr_manifest.json', {'source': str(folder), 'official_counts': counts,
              'split_strategy': 'official image-level; captions never cross split', 'limit': args.limit,
              'splits': result})
    return result


class Vocabulary:
    def __init__(self, captions=None, size=8000, words=None):
        counts = Counter(w for caption in (captions or []) for w in clean(caption).split())
        self.words = words or ['<pad>', '<unk>', '<bos>', '<eos>'] + [w for w, _ in counts.most_common(size - 4)]
        self.index = {w: i for i, w in enumerate(self.words)}

    def encode(self, text, length):
        ids = [2] + [self.index.get(w, 1) for w in clean(text).split()][:length - 2] + [3]
        return ids + [0] * (length - len(ids))

    def decode(self, ids):
        result = []
        for i in ids:
            if i == 3:
                break
            if i > 3:
                result.append(self.words[i])
        return ' '.join(result)


def prepare(args):
    b, f = beans(args), flickr(args)
    result = {'beans': {s: len(b[s]) for s in SPLITS}, 'flickr8k': {s: len(f[s]) for s in SPLITS}}
    save_json(args.output_dir / 'dataset_summary.json', result)
    print(result, flush=True)
