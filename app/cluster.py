# ======================================================
# app/cluster.py — UMAP + Clustering (Updated Version)
# ======================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import AgglomerativeClustering
from matplotlib.lines import Line2D

EMB_PATH = "output/dbn_embeddings.npy"
RESULTS_PATH = "output/match_results.csv"
OUTPUT_PATH = "output/clusters_umap.png"


# ======================================================
# LOAD DATA
# ======================================================
def load_data():
    """Load DBN embeddings + enriched taxonomy results."""

    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError("❌ dbn_embeddings.npy missing. Run taxonomy_matcher.py first!")

    embeddings = np.load(EMB_PATH)

    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError("❌ match_results.csv missing. Run taxonomy_matcher.py first!")

    df = pd.read_csv(RESULTS_PATH)

    # Safe defaults (dashboard will depend on these)
    df["DL_MarineScore"] = df.get("DL_MarineScore", 0.5)
    df["NoveltyScore"] = df.get("NoveltyScore", 0.0)

    labels = df["Organism"].fillna("Unknown").tolist()
    dl_scores = df["DL_MarineScore"].astype(float).tolist()
    novelty = df["NoveltyScore"].astype(float).tolist()

    return embeddings, labels, dl_scores, novelty


# ======================================================
# CLUSTER + PLOT
# ======================================================
def cluster_and_plot(embeddings, labels, dl_scores, novelty_scores, n_clusters=4):

    n = embeddings.shape[0]

    # ------------------------------
    # Case 0: No points
    # ------------------------------
    if n == 0:
        print("⚠️ No embeddings found — skipping clustering.")
        return

    # ------------------------------
    # Case 1: Only one point
    # ------------------------------
    if n == 1:
        print("⚠️ Only 1 sequence — UMAP not meaningful, plotting point directly.")
        reduced = np.array([[0, 0]])  # simple 2D anchor

    else:
        print("📌 Running UMAP on DBN embeddings…")
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)

    # ------------------------------
    # Clustering (only when meaningful)
    # ------------------------------
    if n >= 3:
        k = min(n_clusters, n)
        clusterer = AgglomerativeClustering(n_clusters=k)
        clusters = clusterer.fit_predict(reduced)
    else:
        clusters = np.zeros(n, dtype=int)

    # ------------------------------
    # Color mapping: Marine probability (blue→red)
    # ------------------------------
    cmap = plt.cm.get_cmap("coolwarm")
    colors = [cmap(prob) for prob in dl_scores]

    # ------------------------------
    # Plotting
    # ------------------------------
    plt.figure(figsize=(12, 8))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=colors,
        s=65,
        edgecolor="black",
        alpha=0.90
    )

    # Highlight novelty (NoveltyScore > 0.6)
    for i, (x, y) in enumerate(reduced):
        if novelty_scores[i] > 0.6:
            plt.text(x, y, "⚠️", fontsize=13, color="black")

    # Annotate labels lightly
    for i, label in enumerate(labels):
        plt.annotate(
            label[:20],            # avoid long names
            (reduced[i, 0], reduced[i, 1]),
            fontsize=6,
            alpha=0.55
        )

    # ------------------------------
    # Legend
    # ------------------------------
    legend_elements = [
        Line2D(
            [0], [0], marker='o', color='w',
            label='Low Marine Score (blue)',
            markerfacecolor=cmap(0.1), markersize=10
        ),
        Line2D(
            [0], [0], marker='o', color='w',
            label='High Marine Score (red)',
            markerfacecolor=cmap(0.9), markersize=10
        ),
        Line2D(
            [0], [0], marker='o', color='yellow',
            label='⚠️ Potential Novel Sequence (Novelty > 0.6)',
            markersize=12
        )
    ]

    plt.legend(handles=legend_elements, loc="best")

    plt.title("UMAP Clusters\n(DBN Embeddings + Marine Probability + Novelty Detection)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    # Save
    os.makedirs("output", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"✅ UMAP plot saved → {OUTPUT_PATH}")


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    embeddings, labels, dl_probs, novelty = load_data()
    cluster_and_plot(embeddings, labels, dl_probs, novelty)
