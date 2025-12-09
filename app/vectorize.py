from Bio import SeqIO
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

# --------------------
# PARAMETERS
# --------------------
K = 6
SAMPLE_FASTA = "data/sample1.fasta"
REF_FASTA = "data/strong_reference.fasta"
MATCH_THRESHOLD = 0.05   # minimum similarity to accept match

# Marine keywords (simple version)
MARINE_KEYWORDS = ["marine", "sea", "ocean", "seawater", "deep-sea"]

# --------------------
# KMER FUNCTIONS
# --------------------
def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def fasta_to_vectors(fasta_file, k):
    vectors = []
    labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        kmers = get_kmers(str(record.seq).upper(), k)
        freq = Counter(kmers)
        vectors.append(freq)
        labels.append(record.id)
    return vectors, labels

def to_dense_matrix_pair(sample_vecs, ref_vecs):
    all_kmers = sorted(set(k for vec in (sample_vecs + ref_vecs) for k in vec))

    def vec_to_row(vec):
        return [vec.get(k, 0) for k in all_kmers]

    sample_matrix = np.array([vec_to_row(v) for v in sample_vecs])
    ref_matrix = np.array([vec_to_row(v) for v in ref_vecs])

    return sample_matrix, ref_matrix

# --------------------
# SIMPLE TAXON PARSER
# --------------------
def guess_marine(lineage):
    lineage_l = lineage.lower()
    if any(keyword in lineage_l for keyword in MARINE_KEYWORDS):
        return "Marine"
    return "Non-Marine"

# --------------------
# MAIN
# --------------------
if __name__ == "__main__":

    print("Vectorizing sample...")
    sample_vecs, sample_ids = fasta_to_vectors(SAMPLE_FASTA, K)

    print("Vectorizing reference...")
    ref_vecs, ref_ids = fasta_to_vectors(REF_FASTA, K)

    sample_matrix, ref_matrix = to_dense_matrix_pair(sample_vecs, ref_vecs)

    # Save matrix for clustering
    np.save("vector_matrix.npy", sample_matrix)

    print("\nComputing cosine similarities...")
    sim_matrix = cosine_similarity(sample_matrix, ref_matrix)

    # Load simple taxonomy info from FASTA headers
    # Ex: >ID | Organism | Lineage
    ref_meta = {}
    for record in SeqIO.parse(REF_FASTA, "fasta"):
        header = record.description
        parts = header.split("|")
        if len(parts) >= 3:
            org = parts[1].strip()
            lin = parts[2].strip()
        else:
            org = "Unknown"
            lin = "Unknown"
        ref_meta[record.id] = (org, lin)

    # Write output
    os.makedirs("output", exist_ok=True)
    with open("output/match_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SampleID", "MatchedID", "Organism",
                         "Lineage", "Score", "MarineFlag"])

        print("\nMatches:")
        for i, row in enumerate(sim_matrix):
            best_match_idx = np.argmax(row)
            best_score = row[best_match_idx]
            best_id = ref_ids[best_match_idx]

            organism, lineage = ref_meta.get(best_id, ("Unknown", "Unknown"))
            marine_flag = guess_marine(lineage)

            if best_score >= MATCH_THRESHOLD:
                print(f"{sample_ids[i]} → {best_id} ({organism}) "
                      f"(Score: {best_score:.3f}, {marine_flag})")
            else:
                print(f"{sample_ids[i]} → Unknown (Score: {best_score:.3f})")
                best_id = "Unknown"
                organism = "Unknown"
                lineage = "Unknown"
                marine_flag = "Unknown"

            writer.writerow([sample_ids[i], best_id,
                             organism, lineage, f"{best_score:.3f}", marine_flag])

    print("\n✅ DONE! Output written to output/match_results.csv")
