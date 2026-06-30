"""Standalone transformer training example."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingConfig:
    vocab_size: int = 1000
    seq_len: int = 32
    num_classes: int = 10
    train_samples: int = 512
    val_samples: int = 128
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1


class ToyTextDataset(Dataset):
    """Synthetic dataset for sequence classification."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        num_classes: int,
    ):
        super().__init__()
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        # Use a deterministic label rule so the model has a pattern to learn.
        self.labels = (self.input_ids.sum(dim=1) % num_classes).long()

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.labels[index]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for token sequences."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SimpleTransformerModel(nn.Module):
    """A small transformer encoder for sequence classification."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 10,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run a validation pass and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def train_model(config: TrainingConfig) -> None:
    """Train the transformer on a synthetic classification task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_dataset = ToyTextDataset(
        num_samples=config.train_samples,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
    )
    val_dataset = ToyTextDataset(
        num_samples=config.val_samples,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = SimpleTransformerModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        num_classes=config.num_classes,
        max_len=config.seq_len,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Training on device: {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen_examples = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            seen_examples += labels.size(0)

        train_loss = running_loss / seen_examples
        train_acc = running_correct / seen_examples
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


def main() -> None:
    config = TrainingConfig()
    train_model(config)


if __name__ == "__main__":
    main()
