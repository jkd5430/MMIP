import numpy as np
import torch
from torch.utils.data import TensorDataset

from .common import fit, save_json
from .datasets import beans, BEAN_CLASSES
from .features import extract
from .models import RetrievalModel


def ranking_metrics(query, gallery, query_labels, gallery_labels, k):
    if not len(gallery) or not len(query):
        raise ValueError('Empty retrieval query/gallery')
    k = min(k, len(gallery))
    similarity = query @ gallery.T
    ranks = np.argsort(-similarity, axis=1)[:, :k]
    relevant = gallery_labels[ranks] == query_labels[:, None]
    total_relevant = (gallery_labels[None, :] == query_labels[:, None]).sum(1)
    if np.any(total_relevant == 0):
        raise ValueError('Each query class needs at least one gallery image; increase --limit')
    precision = relevant.cumsum(1) / np.arange(1, k + 1)
    average_precision = (precision * relevant).sum(1) / np.minimum(total_relevant, k)
    return {'precision_at_k': float(relevant.mean()), 'recall_at_k': float((relevant.sum(1) / total_relevant).mean()),
            'hit_at_k': float(relevant.any(1).mean()), 'map_at_k': float(average_precision.mean()),
            'k': k, 'query_count': len(query), 'gallery_count': len(gallery)}, ranks, similarity


def plot_retrieval(queries, gallery, ranks, similarities, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    count, k = min(4, len(queries)), min(5, ranks.shape[1])
    fig, axes = plt.subplots(count, k + 1, figsize=(2.5 * (k + 1), 2.7 * count), squeeze=False)
    for i in range(count):
        selected = [queries[i]] + [gallery[j] for j in ranks[i, :k]]
        for j, row in enumerate(selected):
            with Image.open(row['path']) as image:
                axes[i, j].imshow(image)
            title = 'Query' if j == 0 else 'Top %d / %.3f' % (j, similarities[i, ranks[i, j - 1]])
            axes[i, j].set_title(title + '\n' + BEAN_CLASSES[row['label']], fontsize=9)
            axes[i, j].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args, kind):
    out = args.output_dir / ('q1_' + kind)
    splits = beans(args)
    features = {s: extract(rows, kind, args, 'beans_' + s) for s, rows in splits.items()}
    labels = {s: torch.tensor([r['label'] for r in rows]) for s, rows in splits.items()}
    model = RetrievalModel(features['train'].shape[1], args.hidden_size)
    stats = fit(model, TensorDataset(features['train'], labels['train']),
                TensorDataset(features['validation'], labels['validation']), args, out)
    with torch.inference_mode():
        gallery = model.embed(features['train'].to(args.device)).cpu().numpy()
        query = model.embed(features['test'].to(args.device)).cpu().numpy()
    metrics, ranks, scores = ranking_metrics(query, gallery, labels['test'].numpy(), labels['train'].numpy(), args.top_k)
    metrics.update(stats)
    metrics.update({'backbone': kind, 'task': 'class-relevance retrieval',
                    'relevance': 'same Beans disease/healthy class', 'gallery_split': 'train', 'query_split': 'test',
                    'training_objective': 'cross-entropy class-supervised retrieval embedding',
                    'backbone_frozen': True})
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'config.json', {'kind': kind, 'input_dim': features['train'].shape[1], 'hidden': args.hidden_size})
    save_json(out / 'gallery.json', splits['train'])
    torch.save(torch.from_numpy(gallery), out / 'gallery_features.pt')
    results = []
    for i, row in enumerate(splits['test']):
        results.append({'query': row, 'results': [dict(splits['train'][int(j)], similarity=float(scores[i, j])) for j in ranks[i]]})
    save_json(out / 'retrieval_results.json', results)
    plot_retrieval(splits['test'], splits['train'], ranks, scores, out / 'retrieval_examples.png')
    print(metrics, flush=True)
