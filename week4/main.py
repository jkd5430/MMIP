"""MMIP 第四堂：以專案位置解析路徑，從任何工作目錄均可執行。"""
import argparse
import os
from pathlib import Path

TASKS = ('prepare', 'q2_rnn', 'q2_lstm', 'q3_vit', 'q3_resnet',
         'q4_gru', 'q4_lstm', 'gemini', 'report', 'smoke')


def load_local_environment(root):
    """Load Gemini settings only; shell environment takes precedence."""
    path = Path(root) / '.env'
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, separator, value = line.partition('=')
        name, value = name.strip(), value.strip()
        if separator and name in ('GEMINI_API_KEY', 'GEMINI_MODEL'):
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if value:
                os.environ.setdefault(name, value)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('必須大於 0')
    return number


def main():
    root = Path(__file__).resolve().parent
    load_local_environment(root)
    # Personal Gemini settings embedded at the user's request.
    os.environ['GEMINI_API_KEY'] = 'xxxxxxxxxxxxxx'
    os.environ['GEMINI_MODEL'] = 'gemini-3.5-flash-lite'
    parser = argparse.ArgumentParser(description='MMIP Week 4 / TensorFlow 2.11')
    parser.add_argument('task', nargs='?', default='all',
                        choices=TASKS + ('q2', 'q3', 'q4', 'all'))
    parser.add_argument('--gpu', default=None, help='實體 GPU 編號，例如 4；cpu 表示 CPU')
    parser.add_argument('--epochs', type=positive, default=10)
    parser.add_argument('--warmup-epochs', type=positive, default=3)
    parser.add_argument('--batch-size', type=positive, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=positive, help='每個 split 限制資料量，僅供除錯')
    parser.add_argument('--eval-images', type=positive, default=None,
                        help='Caption 評估張數；預設完整 test split')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Q4：載入已儲存權重，只重新推論、計算 BLEU 與時間，不重訓')
    parser.add_argument('--gemini-samples', type=positive, default=10)
    parser.add_argument('--gemini-model', default=os.getenv('GEMINI_MODEL'),
                        help='指定可用的 Gemini 模型，或設定 GEMINI_MODEL')
    args = parser.parse_args()
    if args.evaluate_only and args.task not in ('q4', 'q4_gru', 'q4_lstm'):
        parser.error('--evaluate-only 僅適用於 q4、q4_gru、q4_lstm')
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.gpu.lower() == 'cpu' else args.gpu
    os.environ.setdefault('KERAS_HOME', str(root / 'data' / 'keras'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    # TensorFlow imports happen only after device selection.
    if args.task == 'report':
        from code.report import run
        run(args)
        return
    if args.task == 'gemini':
        from code.gemini import run
        run(args)
        from code.report import run as report
        report(args)
        return
    from code.common import setup
    setup(args.seed)
    if args.task in ('all', 'q4', 'q4_gru', 'q4_lstm'):
        from code.caption import require_bleu
        require_bleu()
    groups = {
        'q2': ('q2_rnn', 'q2_lstm'),
        'q3': ('q3_vit', 'q3_resnet'),
        'q4': ('q4_gru', 'q4_lstm'),
        'all': ('prepare', 'q2_rnn', 'q2_lstm', 'q3_vit', 'q3_resnet',
                'q4_gru', 'q4_lstm'),
    }
    for task in groups.get(args.task, (args.task,)):
        print('\n===== ' + task + ' =====', flush=True)
        if task == 'prepare':
            from code.datasets import prepare
            prepare(args)
        elif task.startswith('q2_'):
            from code.sentiment import run
            run(args, task.split('_')[1])
        elif task.startswith('q3_'):
            from code.vision import run
            run(args, task.split('_')[1])
        elif task.startswith('q4_'):
            from code.caption import run
            run(args, task.split('_')[1])
        elif task == 'smoke':
            from code.smoke import run
            run(args)
    if args.task == 'all':
        if os.getenv('GEMINI_API_KEY') and args.gemini_model:
            from code.gemini import run
            run(args)
        else:
            print('Gemini 尚未執行：設定 GEMINI_API_KEY 與 GEMINI_MODEL 後執行 main.py gemini。')
    if args.task != 'smoke':
        from code.report import run
        run(args)


if __name__ == '__main__':
    main()
