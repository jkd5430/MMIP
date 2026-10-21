import csv
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def environment(args):
    packages = {}
    for name in ['torch', 'torchvision', 'tensorflow', 'numpy', 'scikit-learn',
                 'matplotlib', 'Pillow', 'transformers', 'nltk', 'bert-score', 'pyarrow']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'packages': packages,
            'device': str(args.device), 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Keep Windows CPU runs responsive, especially GRU inference.
    torch.set_num_threads(min(8, torch.get_num_threads()))
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / 'environment.json', environment(args))
    print('Python:', sys.executable, '| device:', args.device, flush=True)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def loader(dataset, args, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def fit(model, train_data, val_data, args, out, caption=False):
    """Validation-only selection; test is never used for training/stopping."""
    out.mkdir(parents=True, exist_ok=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0) if caption else nn.CrossEntropyLoss()
    best, stale, history = float('inf'), 0, []
    synchronize(args.device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        row = {'epoch': epoch + 1}
        for name, dataset in [('train', train_data), ('validation', val_data)]:
            training = name == 'train'
            model.train(training)
            total_loss, correct, count = 0., 0, 0
            with torch.set_grad_enabled(training):
                for batch in loader(dataset, args, shuffle=training):
                    inputs = [x.to(args.device) for x in batch[:-1]]
                    target = batch[-1].to(args.device)
                    output = model(*inputs)
                    loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    mask = target != 0 if caption else torch.ones_like(target, dtype=torch.bool)
                    n = int(mask.sum())
                    count += n
                    total_loss += float(loss.detach()) * n
                    correct += int(((output.argmax(-1) == target) & mask).sum())
            row[name + '_loss'] = total_loss / max(count, 1)
            row[name + '_accuracy'] = correct / max(count, 1)
        history.append(row)
        print(row, flush=True)
        if row['validation_loss'] < best:
            best, stale = row['validation_loss'], 0
            torch.save(model.state_dict(), out / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    synchronize(args.device)
    duration = time.perf_counter() - start
    model.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True))
    model.eval()
    save_csv(out / 'history.csv', history)
    save_json(out / 'history.json', history)
    plot_history(history, out)
    details = {'epochs_completed': len(history), 'best_epoch': min(history, key=lambda x: x['validation_loss'])['epoch'],
               'training_seconds': duration, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'selection': 'minimum validation loss', 'seed': args.seed,
               'limit': args.limit, 'epochs_requested': args.epochs, 'batch_size': args.batch_size,
               'learning_rate': args.lr, 'patience': args.patience}
    save_json(out / 'training.json', details)
    return details


def plot_history(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ['train', 'validation']:
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot([r['epoch'] for r in rows], [r[key + '_' + metric] for r in rows], label=key)
            ax.set(xlabel='Epoch', ylabel=metric)
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png', dpi=140)
    plt.close(fig)
