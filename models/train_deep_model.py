# models/train_deep_model.py
#
# Train the CNN + DBN model on marine vs non-marine sequences.
#
# Uses:
#   data/train_full/marine.fasta
#   data/train_full/nonmarine.fasta
#
# Saves:
#   models/pretrained/marine_classifier.pt

import os
from Bio import SeqIO
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .deep_model import CNN_DBM_Model, DNAEncoder


# =============================================================
# CONFIG
# =============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train_full")

MARINE_FASTA = os.path.join(DATA_DIR, "marine.fasta")
NONMARINE_FASTA = os.path.join(DATA_DIR, "nonmarine.fasta")

MODEL_DIR = os.path.join(BASE_DIR, "models", "pretrained")
MODEL_PATH = os.path.join(MODEL_DIR, "marine_classifier.pt")

SEQ_LEN = 1500          # max sequence length to encode
BATCH_SIZE = 32
EPOCHS = 10             # you can increase if training is fast
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================
# Dataset
# =============================================================
class FastaDNADataset(Dataset):
    """
    Holds sequences + labels, encodes sequences on-the-fly using DNAEncoder.
    """
    def __init__(self, fasta_paths, labels, seq_len=1500):
        assert len(fasta_paths) == len(labels)

        self.records = []  # (seq_string, label)
        self.encoder = DNAEncoder(seq_len=seq_len)

        for path, label in zip(fasta_paths, labels):
            for rec in SeqIO.parse(path, "fasta"):
                seq = str(rec.seq)
                self.records.append((seq, label))

        print(f"📦 Loaded {len(self.records)} sequences total.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq, label = self.records[idx]
        x = self.encoder.encode(seq)  # (4, seq_len) tensor
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# =============================================================
# Training / Evaluation Helpers
# =============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / total
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    avg_loss = total_loss / total
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


# =============================================================
# MAIN TRAINING LOOP
# =============================================================
def main():
    if not os.path.exists(MARINE_FASTA) or not os.path.exists(NONMARINE_FASTA):
        raise FileNotFoundError(
            "Expected training FASTA files not found. "
            "Run build_training_data.py first."
        )

    print(f"📂 Using data from: {DATA_DIR}")
    print(f"💻 Training on device: {DEVICE}")

    # 1. Build dataset
    dataset = FastaDNADataset(
        fasta_paths=[MARINE_FASTA, NONMARINE_FASTA],
        labels=[1, 0],
        seq_len=SEQ_LEN
    )

    # 2. Train/val split (80/20)
    n_total = len(dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    print(f"🔀 Train samples: {len(train_ds)}")
    print(f"🔍 Val samples:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize model
    model = CNN_DBM_Model(seq_len=SEQ_LEN).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    # 4. Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"\n📅 Epoch {epoch}/{EPOCHS}")
        print(f"   🟢 Train loss: {train_loss:.4f} | acc: {train_acc:.3f}")
        print(f"   🔵 Val   loss: {val_loss:.4f} | acc: {val_acc:.3f}")

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   💾 Saved new best model to {MODEL_PATH} (val_acc={val_acc:.3f})")

    print("\n✅ Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Final model stored at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
