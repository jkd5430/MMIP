"""Offline executable checks; random fixtures are never presented as homework scores."""
import copy

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.models import resnet18
from torchvision.models.vision_transformer import VisionTransformer

from .common import fit, save_json
from .datasets import Vocabulary
from .models import RetrievalModel, MatchModel, CaptionModel
from .retrieval import ranking_metrics
from .multimodal import make_pairs
from .evaluation import align_predictions, caption_metrics


def run(args):
    check_args = copy.copy(args)
    check_args.epochs, check_args.batch_size, check_args.patience = 1, 4, 1
    check_args.limit = 8
    out = args.output_dir / 'smoke'
    features = torch.randn(8, 16)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    fit(RetrievalModel(16, 16), TensorDataset(features, labels), TensorDataset(features, labels), check_args, out / 'retrieval')
    rows = [{'id': 'a', 'captions': ['a dog runs'] * 5}, {'id': 'b', 'captions': ['a cat sleeps'] * 5}]
    vocab = Vocabulary([c for row in rows for c in row['captions']])
    dataset, pairs = make_pairs(rows, features[:2], vocab, 8, args.seed)
    assert all(p['image'] != p['text_image'] for p in pairs if p['label'] == 0)
    model = MatchModel(16, len(vocab.words), 16)
    fit(model, dataset, dataset, check_args, out / 'matching')
    # Packed GRU must ignore added right padding.
    tokens = dataset.tensors[1][:2].to(args.device)
    image = dataset.tensors[0][:2].to(args.device)
    with torch.inference_mode():
        assert torch.allclose(model(image, tokens), model(image, torch.cat([tokens, torch.zeros(2, 4, device=args.device, dtype=torch.long)], 1)), atol=1e-6)
    tokens = torch.tensor([vocab.encode('a dog runs', 8)] * 8)
    decoder = CaptionModel(16, len(vocab.words), 16)
    fit(decoder, TensorDataset(features, tokens[:, :-1], tokens[:, 1:]),
        TensorDataset(features, tokens[:, :-1], tokens[:, 1:]), check_args, out / 'caption', caption=True)
    assert len(decoder.generate(features[:2].to(args.device), 8)) == 2
    with torch.inference_mode():
        cnn = resnet18(weights=None).eval()
        assert cnn(torch.randn(1, 3, 64, 64)).shape == (1, 1000)
        vit = VisionTransformer(image_size=32, patch_size=8, num_layers=1, num_heads=2, hidden_dim=32, mlp_dim=64).eval()
        assert vit(torch.randn(1, 3, 32, 32)).shape == (1, 1000)
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        clip = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=32, intermediate_size=64, projection_dim=16,
                  num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=8)).eval()
        assert clip(pixel_values=torch.randn(1, 3, 32, 32)).image_embeds.shape == (1, 16)
    metrics, _, _ = ranking_metrics(np.eye(2), np.eye(2), np.array([0, 1]), np.array([0, 1]), 1)
    assert metrics['map_at_k'] == 1 and metrics['precision_at_k'] == 1
    a = [{'image': 'a', 'references': ['a dog runs'], 'greedy': 'a dog runs'}]
    b = [{'image': 'other', 'references': ['a cat']}, a[0]]
    assert align_predictions(a, b)[2] == ['a']
    bleu_args = copy.copy(args)
    bleu_args.skip_bertscore = True
    assert caption_metrics(a, 'greedy', bleu_args)['bleu_1'] == 1
    save_json(out / 'checks.json', {'passed': True, 'checks': ['optimizer and checkpoint roundtrip for all 3 task heads',
              'matching negatives exclude own image', 'GRU ignores right padding', 'greedy decoder',
              'CNN and ViT forward', 'transformers CLIP forward', 'known perfect retrieval',
              'Week 4/5 alignment by image ID', 'NLTK BLEU'], 'data': 'random offline fixtures; NOT homework experiment results'})
    print('All offline smoke checks passed.', flush=True)
