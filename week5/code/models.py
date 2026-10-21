import torch
from torch import nn
from torch.nn import functional as F


class RetrievalModel(nn.Module):
    def __init__(self, input_dim, hidden=256, classes=3):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 128))
        self.classifier = nn.Linear(128, classes)

    def embed(self, x):
        return F.normalize(self.projection(x), dim=-1)

    def forward(self, x):
        return self.classifier(self.embed(x))


class MatchModel(nn.Module):
    def __init__(self, image_dim, vocab_size, hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.image_projection = nn.Linear(image_dim, hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Dropout(.2), nn.Linear(hidden, 2))

    def forward(self, image, tokens):
        lengths = (tokens != 0).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(tokens), lengths, batch_first=True, enforce_sorted=False)
        _, state = self.gru(packed)
        text = state[-1]
        image = self.image_projection(image)
        fused = torch.cat([image, text, torch.abs(image - text), image * text], -1)
        return self.classifier(fused)


class CaptionModel(nn.Module):
    def __init__(self, image_dim, vocab_size, hidden=256):
        super().__init__()
        self.initial = nn.Linear(image_dim, hidden)
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, image, tokens):
        state = torch.tanh(self.initial(image)).unsqueeze(0)
        sequence, _ = self.gru(self.embedding(tokens), state)
        return self.output(sequence)

    @torch.inference_mode()
    def generate(self, image, length):
        state = torch.tanh(self.initial(image)).unsqueeze(0)
        token = torch.full((len(image), 1), 2, dtype=torch.long, device=image.device)
        finished = torch.zeros(len(image), dtype=torch.bool, device=image.device)
        generated = []
        for _ in range(length - 1):
            output, state = self.gru(self.embedding(token), state)
            logits = self.output(output[:, -1])
            logits[:, :3] = -float('inf')  # never emit PAD/UNK/BOS
            token = logits.argmax(-1, keepdim=True)
            token[finished] = 3
            generated.append(token)
            finished |= token.squeeze(1) == 3
            if finished.all():
                break
        return torch.cat(generated, 1).cpu().tolist()
