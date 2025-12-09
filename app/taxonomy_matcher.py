# ================================================================
# app/taxonomy_matcher.py — FINAL REGENERATED VERSION
# ================================================================

import os
import sys
import csv
import ssl
from collections import Counter

import numpy as np
from Bio import SeqIO, Entrez
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Import deep model
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from models.deep_model import load_marine_classifier, DNAEncoder

# -----------------------------
# Global config
# -----------------------------
ssl._create_default_https_context = ssl._create_unverified_context

K = 5
SEQ_LEN = 1500  # Must match training

SAMPLE_FASTA = "data/sample1.fasta"
REF_FASTA = "data/reff_10k.fasta"

EMBED_PATH = "output/dbn_embeddings.npy"
RESULT_CSV = "output/match_results.csv"

Entrez.email = "nitian.shubh@gmail.com"
Entrez.api_key = "83f10df2a4a230047ef71cc9d48043c63508"

MARINE_KEYWORDS = [
    "marine", "sea", "ocean", "saltwater",
    "coastal", "pelagic", "benthic", "coral", "reef"
]

# ================================================================
# UTILITIES
# ================================================================

def is_marine(lineage: str) -> str:
    """Heuristic marine check from lineage text."""
    if not isinstance(lineage, str):
        return "Unknown"
    lineage = lineage.lower()
    for kw in MARINE_KEYWORDS:
        if kw in lineage:
            return "Yes"
    return "No"


def get_kmers(seq: str, k: int):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def fasta_to_vectors(fasta_file: str, k: int):
    """Convert fasta → list of k-mer frequency dicts."""
    vectors, labels = [], []
    count = 0

    print(f"\n🔍 Reading FASTA: {fasta_file}")
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).upper()
        kmers = get_kmers(seq, k)

        if not kmers:
            continue

        vectors.append(Counter(kmers))
        labels.append(record.id)

        count += 1
        if count % 500 == 0:
            print(f"   ➤ Processed {count} sequences…")

    print(f"✔ Total sequences read: {count}")
    return vectors, labels


def to_dense_matrix_pair(sample_vecs, ref_vecs):
    """Convert k-mer dictionaries → dense feature vectors."""
    all_kmers = sorted(set(k for vec in (sample_vecs + ref_vecs) for k in vec))

    def convert(vec):
        return [vec.get(k, 0) for k in all_kmers]

    sample_matrix = np.array([convert(v) for v in sample_vecs])
    ref_matrix = np.array([convert(v) for v in ref_vecs])

    return sample_matrix, ref_matrix


# ================================================================
# NCBI TAXONOMY FETCH
# ================================================================

def fetch_taxonomy_details(genbank_id: str):
    """Fetch organism name + lineage."""
    try:
        handle = Entrez.efetch(
            db="nucleotide",
            id=genbank_id,
            rettype="gb",
            retmode="text"
        )
        gb = handle.read()
        handle.close()

        organism = "Unknown"
        lineage_list = []
        capture = False

        for line in gb.splitlines():
            if line.strip().startswith("ORGANISM"):
                organism = line.strip().split("  ")[-1]
                capture = True
                continue

            if capture:
                if line.startswith(" " * 12):
                    lineage_list.append(line.strip())
                else:
                    break

        lineage = " ".join(lineage_list).replace(";", "")
        return organism, lineage

    except:
        return "Unknown", "Unknown"


# ================================================================
# DEEP MODEL: MARINE CLASSIFIER + EMBEDDINGS
# ================================================================

def run_deep_model_on_fasta(fasta_path: str):
    """Runs CNN+DBN model → marine prob + embeddings."""
    print("\n📌 Step 3: Deep model scoring…")

    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        print("⚠ No sequences found for deep model!")
        return np.array([]), np.zeros((0, 256))

    encoder = DNAEncoder(seq_len=SEQ_LEN)
    model = load_marine_classifier(
        path=os.path.join(PROJECT_ROOT, "models", "pretrained", "marine_classifier.pt"),
        seq_len=SEQ_LEN,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch = []
    for rec in records:
        batch.append(encoder.encode(str(rec.seq)))

    X = torch.stack(batch).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1]       # marine prob
        embeddings = model.get_embedding(X)

    probs = probs.cpu().numpy()
    embeddings = embeddings.cpu().numpy()

    os.makedirs("output", exist_ok=True)
    np.save(EMBED_PATH, embeddings)
    print(f"✅ Saved embeddings → {EMBED_PATH}")

    return probs, embeddings


# ================================================================
# NOVELTY DETECTION
# ================================================================

def compute_novelty_scores(embeddings: np.ndarray):
    """Novelty based on k-nearest embedding distance."""
    n = embeddings.shape[0]
    if n <= 1:
        return np.zeros(n)

    # Pairwise distances
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    np.fill_diagonal(dist, np.inf)
    k = min(5, n - 1)

    mean_knn = np.sort(dist, axis=1)[:, :k].mean(axis=1)
    d_min, d_max = mean_knn.min(), mean_knn.max()

    if d_max == d_min:
        return np.zeros(n)

    return (mean_knn - d_min) / (d_max - d_min)


def classify_novelty(score: float):
    if score < 0.3:
        return "Known"
    if score < 0.6:
        return "Divergent strain"
    return "Potential novel species"


def habitat_consensus(dl_prob: float, ref_flag: str):
    if dl_prob >= 0.7 and ref_flag == "Yes":
        return "Marine (DL + ref agree)"
    if dl_prob >= 0.7 and ref_flag == "No":
        return "Marine-like (DL only)"
    if dl_prob <= 0.3 and ref_flag == "No":
        return "Non-marine (DL + ref agree)"
    if dl_prob <= 0.3 and ref_flag == "Yes":
        return "Non-marine-like (DL only)"
    return "Ambiguous"


def confidence_score(hybrid: float, novelty: float):
    return hybrid * (1 - novelty)


# ================================================================
# MAIN PIPELINE
# ================================================================

if __name__ == "__main__":

    # -----------------------------
    # Step 1: Sample k-mers
    # -----------------------------
    print("\n📌 Step 1: K-mer vectorization (sample)…")
    sample_vecs, sample_ids = fasta_to_vectors(SAMPLE_FASTA, K)

    # -----------------------------
    # Step 2: Reference k-mers
    # -----------------------------
    print("\n📌 Step 2: K-mer vectorization (reference)…")
    ref_vecs, ref_ids = fasta_to_vectors(REF_FASTA, K)

    # -----------------------------
    # Step 3a: Dense matrices
    # -----------------------------
    print("\n📌 Step 3a: Constructing feature matrices…")
    sample_matrix, ref_matrix = to_dense_matrix_pair(sample_vecs, ref_vecs)
    np.save("output/vector_matrix.npy", sample_matrix)

    # -----------------------------
    # Step 3b: Deep model scoring
    # -----------------------------
    dl_probs, embeddings = run_deep_model_on_fasta(SAMPLE_FASTA)

    if len(dl_probs) != len(sample_ids):
        print("⚠ Deep model mismatch → using fallback values.")
        dl_probs = np.full(len(sample_ids), 0.5)
        embeddings = np.zeros((len(sample_ids), 256))

    novelty_scores = compute_novelty_scores(embeddings)

    # -----------------------------
    # Step 4: Cosine similarity
    # -----------------------------
    print("\n📌 Step 4: Cosine similarity search…")
    sim_matrix = cosine_similarity(sample_matrix, ref_matrix)

    # -----------------------------
    # Step 5: Write outputs
    # -----------------------------
    print("\n📌 Step 5: Saving enriched results…")
    os.makedirs("output", exist_ok=True)

    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SampleID",
            "MatchedID",
            "Organism",
            "Lineage",
            "ReferenceMarineFlag",
            "RawScore",
            "DL_MarineScore",
            "HybridScore",
            "NoveltyScore",
            "NoveltyType",
            "HabitatConsensus",
            "ConfidenceScore",
        ])

        for i, row in enumerate(sim_matrix):
            best = np.argmax(row)
            raw = float(row[best])
            genbank = ref_ids[best]

            organism, lineage = fetch_taxonomy_details(genbank)
            ref_flag = is_marine(lineage)
            dl_prob = float(dl_probs[i])

            hybrid = 0.5 * raw + 0.5 * dl_prob
            nov = float(novelty_scores[i])
            nov_type = classify_novelty(nov)
            hab = habitat_consensus(dl_prob, ref_flag)
            conf = confidence_score(hybrid, nov)

            print(f"{sample_ids[i]} → {genbank} | "
                  f"Raw={raw:.3f} | DL={dl_prob:.3f} | "
                  f"Hybrid={hybrid:.3f} | Novel={nov:.3f} ({nov_type})")

            writer.writerow([
                sample_ids[i],
                genbank,
                organism,
                lineage,
                ref_flag,
                f"{raw:.3f}",
                f"{dl_prob:.3f}",
                f"{hybrid:.3f}",
                f"{nov:.3f}",
                nov_type,
                hab,
                f"{conf:.3f}",
            ])

    print(f"\n✅ DONE! Saved enriched results → {RESULT_CSV}")
    print(f"   Embeddings saved → {EMBED_PATH}")
