import torch
from torch.utils.data import TensorDataset

from .common import fit, loader, save_json
from .datasets import flickr, Vocabulary
from .features import extract
from .models import CaptionModel


def caption_dataset(rows, features, vocabulary, length):
    images, tokens = [], []
    for i, row in enumerate(rows):
        for caption in row['captions']:
            images.append(i)
            tokens.append(vocabulary.encode(caption, length))
    tokens = torch.tensor(tokens)
    return TensorDataset(features[images], tokens[:, :-1], tokens[:, 1:])


def run(args):
    out = args.output_dir / 'q3_clip_gru'
    splits = flickr(args)
    vocabulary = Vocabulary([c for row in splits['train'] for c in row['captions']], args.vocab_size)
    features = {split: extract(rows, 'clip', args, 'flickr_' + split) for split, rows in splits.items()}
    model = CaptionModel(features['train'].shape[1], len(vocabulary.words), args.hidden_size)
    stats = fit(model, caption_dataset(splits['train'], features['train'], vocabulary, args.max_length),
                caption_dataset(splits['validation'], features['validation'], vocabulary, args.max_length),
                args, out, caption=True)
    save_json(out / 'vocabulary.json', vocabulary.words)
    save_json(out / 'config.json', {'image_dim': features['train'].shape[1], 'hidden': args.hidden_size, 'length': args.max_length,
              'encoder': 'openai/clip-vit-base-patch32', 'encoder_frozen': True, 'decoder': 'GRU', 'decoding': 'greedy'})
    predictions = []
    for (batch,) in loader(TensorDataset(features['test']), args):
        generated = model.generate(batch.to(args.device), args.max_length)
        for ids in generated:
            row = splits['test'][len(predictions)]
            predictions.append({'image': row['id'], 'image_path': row['path'],
                                'references': row['captions'], 'greedy': vocabulary.decode(ids)})
    save_json(out / 'predictions.json', predictions)
    from .evaluation import caption_metrics
    metrics = caption_metrics(predictions, 'greedy', args)
    metrics.update(stats)
    metrics['encoder'] = 'CLIP ViT-B/32 (frozen)'
    metrics['decoder'] = 'trained GRU'
    save_json(out / 'metrics.json', metrics)
    plot_examples(predictions, out / 'caption_examples.png')
    print(metrics, flush=True)


def plot_examples(rows, path):
    import textwrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    count = min(6, len(rows))
    fig, axes = plt.subplots(count, 2, figsize=(12, count * 2.8), squeeze=False, gridspec_kw={'width_ratios': [1, 2]})
    for i, row in enumerate(rows[:count]):
        with Image.open(row['image_path']) as image:
            axes[i, 0].imshow(image)
        text = 'Generated: ' + row['greedy'] + '\n\nReference: ' + row['references'][0]
        axes[i, 1].text(.02, .9, '\n'.join(textwrap.fill(line, 70) for line in text.split('\n')), va='top', fontsize=10)
        for ax in axes[i]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
