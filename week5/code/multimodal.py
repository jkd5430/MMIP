import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import TensorDataset

from .common import fit, loader, save_json
from .datasets import flickr, Vocabulary
from .features import extract
from .models import MatchModel


def make_pairs(rows, features, vocabulary, length, seed):
    """Balanced positives/negatives, fixed by split and seed for backbone comparison."""
    if len(rows) < 2:
        raise ValueError('Flickr matching requires >= 2 images per split')
    rng = np.random.default_rng(seed)
    pairs, images, tokens, labels = [], [], [], []
    for i, row in enumerate(rows):
        own_text = set(row['captions'])
        for caption in row['captions']:
            # Exclude same image AND verbatim own captions; other false negatives remain possible.
            for _ in range(1000):
                other = int(rng.integers(len(rows)))
                negative = rows[other]['captions'][int(rng.integers(5))]
                if other != i and negative not in own_text:
                    break
            else:
                raise ValueError('Cannot sample a distinct negative caption')
            for text, label, text_id in [(caption, 1, row['id']), (negative, 0, rows[other]['id'])]:
                images.append(i)
                tokens.append(vocabulary.encode(text, length))
                labels.append(label)
                pairs.append({'image': row['id'], 'text_image': text_id, 'text': text, 'label': label})
    return TensorDataset(features[images], torch.tensor(tokens), torch.tensor(labels)), pairs


@torch.inference_mode()
def predict(model, dataset, args):
    probabilities = []
    model.eval()
    for image, text, _ in loader(dataset, args):
        probabilities.extend(model(image.to(args.device), text.to(args.device)).softmax(-1)[:, 1].cpu().tolist())
    return np.array(probabilities)


def run(args, kind):
    out = args.output_dir / ('q2_' + kind + '_gru')
    splits = flickr(args)
    vocabulary = Vocabulary([c for r in splits['train'] for c in r['captions']], args.vocab_size)
    datasets, pairs = {}, {}
    for offset, (split, rows) in enumerate(splits.items()):
        features = extract(rows, kind, args, 'flickr_' + split)
        datasets[split], pairs[split] = make_pairs(rows, features, vocabulary, args.max_length, args.seed + offset)
    image_dim = datasets['train'].tensors[0].shape[1]
    model = MatchModel(image_dim, len(vocabulary.words), args.hidden_size)
    stats = fit(model, datasets['train'], datasets['validation'], args, out)
    probability = predict(model, datasets['test'], args)
    target = datasets['test'].tensors[-1].numpy()
    predicted = (probability >= .5).astype(int)
    report = classification_report(target, predicted, labels=[0, 1], target_names=['not_match', 'match'], output_dict=True, zero_division=0)
    metrics = dict(stats, accuracy=float(accuracy_score(target, predicted)), macro_f1=float(report['macro avg']['f1-score']),
                   auc=float(roc_auc_score(target, probability)), classification_report=report,
                   confusion_matrix=confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
                   test_pairs=len(target), backbone=kind + '+GRU', backbone_frozen=True,
                   caveat='Random mismatches are weak negatives; a different image caption may still describe the query.')
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'vocabulary.json', vocabulary.words)
    save_json(out / 'config.json', {'kind': kind, 'image_dim': image_dim, 'hidden': args.hidden_size, 'length': args.max_length})
    save_json(out / 'pair_manifest.json', pairs)
    save_json(out / 'predictions.json', [dict(row, probability_match=float(p), prediction=int(y))
              for row, p, y in zip(pairs['test'], probability, predicted)])
    plot_classification(target, probability, out)
    print({k: metrics[k] for k in ['accuracy', 'macro_f1', 'auc']}, flush=True)


def plot_classification(target, probability, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(target, probability >= .5, display_labels=['not_match', 'match'], ax=axes[0], colorbar=False)
    RocCurveDisplay.from_predictions(target, probability, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out / 'classification.png', dpi=140)
    plt.close(fig)
