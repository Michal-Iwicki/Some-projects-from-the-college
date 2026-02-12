import torch
import torch.nn as nn
import torchaudio
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm


class Mel_transformer(nn.Module):
    def __init__(self, num_classes=30, n_mels=64, transformer_dim=256, num_heads=4, num_layers=4):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Patch projection to transformer dim
        self.patch_proj = nn.Linear(n_mels * 64, transformer_dim)  # 64: CNN output channels

        self.pos_embedding = nn.Parameter(torch.randn(1, 101, transformer_dim))  # 101: ~1s/10ms

        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 16000)
        x = self.melspec(x)  # (B, n_mels, T)
        x = self.db(x)  # (B, n_mels, T)

        x = self.cnn(x)  # (B, 64, n_mels, T)
        B, C, H, W = x.shape
        x = x.view(B, C * H, W)  # (B, C*H, T)
        x = x.permute(0, 2, 1)  # (B, T, C*H)

        x = self.patch_proj(x)  # (B, T, transformer_dim)
        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer_encoder(x)  # (B, T, transformer_dim)
        x = x.mean(dim=1)  # (B, transformer_dim)

        return self.cls_head(x)  # (B, num_classes)


class RawAudioTransformer(nn.Module):
    def __init__(self, num_classes=30, conv_channels=64, transformer_dim=128, nhead=4, num_layers=4):
        super(RawAudioTransformer, self).__init__()

        # 1D CNN to extract local patterns and reduce sequence length
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, transformer_dim, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU()
        )

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2000, transformer_dim))  # assuming ~2000 steps after conv

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=512,
                                                   dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # average over time dimension
            nn.Flatten(),
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)  # -> (batch_size, transformer_dim, seq_len)
        x = x.permute(0, 2, 1)  # -> (batch_size, seq_len, transformer_dim)

        seq_len = x.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb

        x = self.transformer(x)  # -> (batch_size, seq_len, transformer_dim)
        x = x.permute(0, 2, 1)  # -> (batch_size, transformer_dim, seq_len)
        x = self.classifier(x)  # -> (batch_size, num_classes)
        return x


def train_transformer(model, optimizer, train_loader, val_loader, num_epochs=20, patience=3, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            loop.set_postfix(loss=loss.item())

        train_acc = correct / total
        avg_loss = total_loss / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)

        val_acc = val_correct / val_total

        if verbose:
            print(f"Epoch {epoch + 1}: Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
    if verbose:
        print(f"Best Val Acc: {best_val_acc:.4f}")
        
    model.load_state_dict(best_model_state)

    return 


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # (B, num_classes)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return round(avg_loss, 4), round(accuracy, 4),
