from pathlib import Path

import torch
from torch.nn import functional as F

from .common import read_json, save_json
from .datasets import Vocabulary
from .features import extract
from .models import RetrievalModel, MatchModel, CaptionModel


@torch.inference_mode()
def run(args):
    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(str(image))
    rows = [{'id': str(image), 'path': str(image)}]
    if args.task == 'retrieve':
        out = args.output_dir / ('q1_' + args.backbone)
        config = read_json(out / 'config.json')
        model = RetrievalModel(config['input_dim'], config['hidden']).to(args.device)
        features = extract(rows, config['kind'], args, 'inference')
        model.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True))
        model.eval()
        query = model.embed(features.to(args.device)).cpu()[0]
        gallery = torch.load(out / 'gallery_features.pt', map_location='cpu', weights_only=True)
        items = read_json(out / 'gallery.json')
        scores = gallery @ query
        # A user can query a gallery file; exclude that exact path from its own results.
        eligible = [i for i, row in enumerate(items) if Path(row['path']).resolve() != image]
        selected = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)[:args.top_k]
        result = {'query': str(image), 'results': [dict(items[i], similarity=float(scores[i])) for i in selected]}
    elif args.task == 'match':
        out = args.output_dir / ('q2_' + args.backbone + '_gru')
        config = read_json(out / 'config.json')
        vocabulary = Vocabulary(words=read_json(out / 'vocabulary.json'))
        model = MatchModel(config['image_dim'], len(vocabulary.words), config['hidden']).to(args.device)
        features = extract(rows, config['kind'], args, 'inference')
        model.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True))
        model.eval()
        tokens = torch.tensor([vocabulary.encode(args.text, config['length'])], device=args.device)
        probability = float(F.softmax(model(features.to(args.device), tokens), -1)[0, 1])
        result = {'image': str(image), 'text': args.text, 'probability_match': probability, 'prediction': 'match' if probability >= .5 else 'not_match'}
    else:
        out = args.output_dir / 'q3_clip_gru'
        config = read_json(out / 'config.json')
        vocabulary = Vocabulary(words=read_json(out / 'vocabulary.json'))
        model = CaptionModel(config['image_dim'], len(vocabulary.words), config['hidden']).to(args.device)
        features = extract(rows, 'clip', args, 'inference')
        model.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True))
        model.eval()
        result = {'image': str(image), 'caption': vocabulary.decode(model.generate(features.to(args.device), config['length'])[0])}
    save_json(args.output_dir / 'inference' / (args.task + '.json'), result)
    print(result, flush=True)
