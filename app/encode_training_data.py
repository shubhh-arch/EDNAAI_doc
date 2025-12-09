# app/encode_training_data.py
#
# Step 1: Convert marine / non-marine FASTA sequences into
# fixed-length k-mer vectors and save as NumPy arrays.
#
# Uses:
#   data/train_full/marine.fasta
#   data/train_full/nonmarine.fasta
#
# Outputs:
#   data/train_full/X_train.npy
#   data/train_full/y_train.npy
#   data/train_full/X_val.npy
#   data/train_full/y_val.npy

from Bio import SeqIO
import numpy as np
import os
import random

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = os.path.join("data", "train_full")
MARINE_FASTA = os.path.join(DATA_DIR, "marine.fasta")
NONMARINE_FASTA = os.path.join(DATA_DIR, "nonmarine.fasta")

K = 6               # k-mer length
ALPHABET = "ACGT"   # allowed bases
N_KMERS = 4 ** K    # = 4096 features


# -----------------------------
# K-MER ENCODING UTILITIES
# -----------------------------
char_to_idx = {c: i for i, c in enumerate(ALPHABET)}

def kmer_index(kmer: str) -> int:
    """
    Map a k-mer like 'ACGTAA' to an integer index in [0, 4^k - 1]
    using base-4 encoding (A=0, C=1, G=2, T=3).
    """
    idx = 0
    for ch in kmer:
        idx *= 4
        idx += char_to_idx.get(ch, 0)  # unknowns treated as 'A'
    return idx


def sequence_to_kmer_vector(seq: str, k: int = K) -> np.ndarray:
    """
    Convert a DNA sequence string into a normalized k-mer frequency vector
    of shape (4^k,).
    """
    seq = seq.upper()
    vec = np.zeros(N_KMERS, dtype=np.float32)

    if len(seq) < k:
        return vec

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        # skip k-mers with N or non-ACGT characters
        if any(ch not in ALPHABET for ch in kmer):
            continue
        idx = kmer_index(kmer)
        vec[idx] += 1.0

    total = vec.sum()
    if total > 0:
        vec /= total

    return vec


# -----------------------------
# LOAD FASTA AND BUILD DATASET
# -----------------------------
def load_fasta_as_vectors(path: str, label: int):
    """
    Read all sequences from a FASTA and return (X, y) where:
      - X is a list of 4096-dim vectors
      - y is a list of labels (all = label)
    """
    X = []
    y = []
    count = 0

    print(f"🔍 Reading {path} (label={label})")
    for rec in SeqIO.parse(path, "fasta"):
        seq = str(rec.seq)
        vec = sequence_to_kmer_vector(seq, K)
        X.append(vec)
        y.append(label)
        count += 1

        if count % 1000 == 0:
            print(f"   processed {count} sequences...")

    print(f"✔ Total {count} sequences from {os.path.basename(path)}")
    return X, y


def main():
    if not os.path.exists(MARINE_FASTA) or not os.path.exists(NONMARINE_FASTA):
        raise FileNotFoundError(
            f"Expected {MARINE_FASTA} and {NONMARINE_FASTA} to exist. "
            "Run build_training_data.py first."
        )

    # Load marine (label 1) and non-marine (label 0)
    X_marine, y_marine = load_fasta_as_vectors(MARINE_FASTA, label=1)
    X_non, y_non = load_fasta_as_vectors(NONMARINE_FASTA, label=0)

    X = np.vstack(X_marine + X_non)
    y = np.array(y_marine + y_non, dtype=np.int64)

    print("\n📦 Dataset shapes before shuffle:")
    print("X:", X.shape)
    print("y:", y.shape)

    # -------------------------
    # SHUFFLE
    # -------------------------
    idx = list(range(len(y)))
    random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # -------------------------
    # TRAIN / VAL SPLIT
    # -------------------------
    val_ratio = 0.2
    n_total = len(y)
    n_val = int(n_total * val_ratio)

    X_val = X[:n_val]
    y_val = y[:n_val]
    X_train = X[n_val:]
    y_train = y[n_val:]

    print("\n📊 Final split:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)

    # -------------------------
    # SAVE TO DISK
    # -------------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)

    print("\n✅ Saved encoded datasets to:")
    print("  ", os.path.join(DATA_DIR, "X_train.npy"))
    print("  ", os.path.join(DATA_DIR, "y_train.npy"))
    print("  ", os.path.join(DATA_DIR, "X_val.npy"))
    print("  ", os.path.join(DATA_DIR, "y_val.npy"))
    print("\nNow we can train CNN+DBN on these features.")


if __name__ == "__main__":
    main()
