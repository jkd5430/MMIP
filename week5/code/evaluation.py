from .common import read_json, save_json, save_csv
from .datasets import clean, week4_candidates


def caption_metrics(rows, method, args, scorer=None):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    references = [[clean(text).split() for text in row['references']] for row in rows]
    hypotheses = [clean(row[method]).split() for row in rows]
    result = {'samples': len(rows), 'decoding': method,
              'bleu_definition': 'NLTK corpus BLEU, scale 0-1, smoothing method1; all 5 references',
              'bertscore_model': args.bertscore_model, 'bertscore_layers': args.bertscore_layers,
              'bertscore_rescale_with_baseline': False, 'bertscore_idf': False}
    for n in range(1, 5):
        result['bleu_' + str(n)] = float(corpus_bleu(references, hypotheses, weights=tuple([1. / n] * n),
                                                    smoothing_function=SmoothingFunction().method1))
    if args.skip_bertscore:
        result['bertscore_status'] = 'skipped; Q3 advanced comparison incomplete'
    else:
        if scorer is None:
            scorer = make_scorer(args)
        precision, recall, f1 = scorer.score([row[method] for row in rows], [row['references'] for row in rows], batch_size=args.batch_size)
        result.update({'bertscore_precision': float(precision.mean()), 'bertscore_recall': float(recall.mean()),
                       'bertscore_f1': float(f1.mean()), 'bertscore_hash': scorer.hash, 'bertscore_status': 'completed'})
    return result


def make_scorer(args):
    from bert_score import BERTScorer
    # BERTScorer itself does not expose local_files_only; transformers respects HF_HUB_OFFLINE.
    return BERTScorer(model_type=args.bertscore_model, num_layers=args.bertscore_layers,
                      device=str(args.device), lang='en', idf=False, rescale_with_baseline=False)


def align_predictions(current, previous):
    def index(rows):
        result = {row['image']: row for row in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate image IDs in predictions')
        return result
    now, old = index(current), index(previous)
    common = sorted(set(now) & set(old))
    if not common:
        raise ValueError('No common image IDs between Week 4 and Week 5')
    for name in common:
        # Match the exact reference sets, ignoring harmless punctuation/ordering differences.
        if sorted(clean(t) for t in now[name]['references']) != sorted(clean(t) for t in old[name]['references']):
            raise ValueError('Different Week 4/5 reference captions for ' + name)
    return [now[n] for n in common], [old[n] for n in common], common


def compare(args):
    current_path = args.output_dir / 'q3_clip_gru' / 'predictions.json'
    if not current_path.exists():
        raise FileNotFoundError('Run q3 first: ' + str(current_path))
    source = None
    for folder in week4_candidates(args):
        candidate = folder / 'results' / 'q4_gru' / 'predictions.json'
        if candidate.exists():
            source = candidate
            break
    if source is None:
        raise FileNotFoundError('Week 4 q4_gru/predictions.json missing; specify --week4-dir')
    current = read_json(current_path)
    previous = read_json(source)
    current, previous, common = align_predictions(current, previous)
    scorer = None if args.skip_bertscore else make_scorer(args)
    rows = []
    for name, data, method in [('week5_clip_gru', current, 'greedy'), ('week4_mobilenetv2_gru', previous, 'greedy'),
                               ('week4_mobilenetv2_gru', previous, 'beam3')]:
        if not all(method in row for row in data):
            continue
        metrics = caption_metrics(data, method, args, scorer)
        rows.append(dict(model=name, **metrics))
    out = args.output_dir / 'comparison'
    save_json(out / 'caption_comparison.json', {'week4_source': str(source), 'image_ids': common,
              'comparison_scope': 'same image IDs and references, same scorer/config; encoder, decoder and training settings may differ',
              'subset': len(common) != 1000, 'results': rows})
    save_csv(out / 'caption_comparison.csv', rows)
    print(rows, flush=True)
