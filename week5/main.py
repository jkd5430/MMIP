"""MMIP Week 5: run with the existing allen_liver Python 3.8 environment."""
import argparse
import os
from pathlib import Path


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return value


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='MMIP Week 5 / PyTorch / allen_liver')
    parser.add_argument('task', nargs='?', default='all', choices=[
        'all', 'prepare', 'q1', 'q1_cnn', 'q1_vit', 'q2', 'q2_cnn_gru',
        'q2_vit_gru', 'q3', 'compare', 'report', 'smoke', 'environment',
        'retrieve', 'match', 'caption'])
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0')
    parser.add_argument('--epochs', type=positive, default=10)
    parser.add_argument('--batch-size', type=positive, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=positive, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=positive, help='images per split; debug only')
    parser.add_argument('--output-dir', type=Path, default=root / 'results')
    parser.add_argument('--week4-dir', type=Path, help='Week 4 project containing results/q4_gru')
    parser.add_argument('--flickr-dir', type=Path, help='existing Flickr8k directory')
    parser.add_argument('--beans-dir', type=Path, help='existing Beans parquet directory')
    parser.add_argument('--max-length', type=positive, default=32)
    parser.add_argument('--vocab-size', type=positive, default=8000)
    parser.add_argument('--hidden-size', type=positive, default=256)
    parser.add_argument('--top-k', type=positive, default=5)
    parser.add_argument('--skip-bertscore', action='store_true', help='debug only; advanced Q3 remains incomplete')
    parser.add_argument('--bertscore-model', default='distilbert-base-uncased')
    parser.add_argument('--bertscore-layers', type=positive, default=5)
    parser.add_argument('--image', type=Path, help='new image for retrieve/match/caption')
    parser.add_argument('--text', help='candidate text for match')
    parser.add_argument('--backbone', choices=['cnn', 'vit'], default='cnn')
    args = parser.parse_args()
    if args.lr <= 0 or args.max_length < 3:
        parser.error('--lr must be positive; --max-length must be >= 3')
    if args.task in ('retrieve', 'match', 'caption') and not args.image:
        parser.error('inference requires --image')
    if args.task == 'match' and not args.text:
        parser.error('match requires --text')
    # Select PyTorch only, so installed TensorFlow/Keras is never imported by transformers.
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('HF_HOME', str(root / 'data' / 'huggingface'))
    os.environ.setdefault('TORCH_HOME', str(root / 'data' / 'torch'))
    from code.common import setup, save_json, environment
    setup(args)
    if args.task == 'environment':
        save_json(args.output_dir / 'environment.json', environment(args))
        return
    if args.task == 'report':
        from code.report import run
        run(args)
        return
    if args.task == 'smoke':
        from code.smoke import run
        run(args)
        return
    if args.task in ('retrieve', 'match', 'caption'):
        from code.inference import run
        run(args)
        return
    groups = {'all': ['prepare', 'q1_cnn', 'q1_vit', 'q2_cnn_gru', 'q2_vit_gru', 'q3', 'compare'],
              'q1': ['q1_cnn', 'q1_vit'], 'q2': ['q2_cnn_gru', 'q2_vit_gru']}
    for task in groups.get(args.task, [args.task]):
        print('\n===== %s =====' % task, flush=True)
        if task == 'prepare':
            from code.datasets import prepare
            prepare(args)
        elif task.startswith('q1_'):
            from code.retrieval import run
            run(args, task.split('_')[1])
        elif task.startswith('q2_'):
            from code.multimodal import run
            run(args, task.split('_')[1])
        elif task == 'q3':
            from code.caption import run
            run(args)
        elif task == 'compare':
            from code.evaluation import compare
            compare(args)
    from code.report import run
    run(args)


if __name__ == '__main__':
    main()
