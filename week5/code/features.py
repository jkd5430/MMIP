"""Frozen pretrained image backbones; cache features to fit an 11 GB GPU."""
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from .common import DATA, loader, synchronize


def ensure_torchvision_weights(weights):
    """Use requests' verified CA bundle: this PC's Windows SSL store is malformed."""
    from .datasets import download
    filename = Path(urlparse(weights.url).path).name
    target = Path(torch.hub.get_dir()) / 'checkpoints' / filename
    if not target.exists():
        download(weights.url, target)
    expected = filename.rsplit('-', 1)[-1].split('.')[0]
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if not digest.hexdigest().startswith(expected):
        raise ValueError('Pretrained weight SHA256 mismatch: ' + str(target))


def build_backbone(kind, device):
    if kind == 'cnn':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        ensure_torchvision_weights(weights)
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        transform, dim, identity = weights.transforms(), 512, 'resnet18-imagenet1k-v1'
    elif kind == 'vit':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        ensure_torchvision_weights(weights)
        model = models.vit_b_16(weights=weights)
        model.heads = nn.Identity()
        transform, dim, identity = weights.transforms(), 768, 'vit-b16-imagenet1k-v1'
    elif kind == 'clip':
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        identity = 'openai/clip-vit-base-patch32'
        try:
            model = CLIPVisionModelWithProjection.from_pretrained(identity, local_files_only=True)
            processor = CLIPImageProcessor.from_pretrained(identity, local_files_only=True)
        except OSError:
            model = CLIPVisionModelWithProjection.from_pretrained(identity)
            processor = CLIPImageProcessor.from_pretrained(identity)
        transform = lambda image: processor(images=image, return_tensors='pt')['pixel_values'][0]
        dim = model.config.projection_dim
    else:
        raise ValueError('Unknown backbone: ' + kind)
    model.eval().to(device)
    model.requires_grad_(False)
    return model, transform, dim, identity


class ImageDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]['path']) as image:
            return self.transform(image.convert('RGB'))


@torch.inference_mode()
def extract(rows, kind, args, namespace):
    # Include IDs, paths, sizes and mtimes so changed inputs/subsets cannot reuse stale features.
    signature = [(r['id'], r['path'], Path(r['path']).stat().st_size,
                  Path(r['path']).stat().st_mtime_ns) for r in rows]
    digest = hashlib.sha256(repr((kind, signature, 'features-v1')).encode()).hexdigest()[:20]
    cache = DATA / 'features' / (namespace + '_' + kind + '_' + digest + '.pt')
    if cache.exists():
        return torch.load(cache, map_location='cpu', weights_only=True)
    model, transform, _, identity = build_backbone(kind, args.device)
    chunks = []
    dataset = ImageDataset(rows, transform)
    synchronize(args.device)
    for i, batch in enumerate(loader(dataset, args)):
        batch = batch.to(args.device)
        value = model(pixel_values=batch).image_embeds if kind == 'clip' else model(batch)
        chunks.append(value.cpu().float())
        if i % 25 == 0:
            print('%s %s: %d/%d images' % (namespace, identity, min((i + 1) * args.batch_size, len(rows)), len(rows)), flush=True)
    features = torch.cat(chunks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache)
    del model
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return features
