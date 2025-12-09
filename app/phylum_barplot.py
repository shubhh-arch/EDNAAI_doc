# ===========================================================
# app/phylum_barplot.py — Improved Multi-Level Biodiversity Plots
# ===========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "output/match_results.csv"
OUT_DIR = "output"


# -----------------------------------------------------------
# SAFE TAXON PARSER (Phylum → Genus)
# -----------------------------------------------------------
def extract_rank(lineage: str, index: int):
    """
    Extract a taxonomic rank from lineage by index.
    Handles:
        - Unknown / NaN
        - Short lineage
        - Multi-word names
    """
    if not isinstance(lineage, str):
        return "Unknown"

    parts = lineage.split()
    if len(parts) > index:
        return parts[index]
    return "Unknown"


def parse_all_ranks(df):
    """
    Extracts multiple hierarchical taxonomic levels safely:

    Index positions typically align as:
        0 = Superkingdom
        1 = Kingdom
        2 = Phylum
        3 = Class
        4 = Order
        5 = Family
        6 = Genus (approx)
    """

    df["Phylum"] = df["Lineage"].apply(lambda x: extract_rank(x, 2))
    df["Class"] = df["Lineage"].apply(lambda x: extract_rank(x, 3))
    df["Order"] = df["Lineage"].apply(lambda x: extract_rank(x, 4))
    df["Family"] = df["Lineage"].apply(lambda x: extract_rank(x, 5))
    df["Genus"] = df["Lineage"].apply(lambda x: extract_rank(x, 6))

    return df


# -----------------------------------------------------------
# GENERIC BARPLOT GENERATOR
# -----------------------------------------------------------
def make_barplot(counts, title, filename, color="#2E7D32"):
    if len(counts) == 0:
        print(f"⚠️ Skipping empty plot: {title}")
        return

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar", color=color)
    plt.title(title)
    plt.ylabel("Read Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()

    print(f"✅ Saved {title} → {filename}")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError("❌ match_results.csv missing — run taxonomy_matcher.py first!")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    # Ensure NoveltyScore exists
    df["NoveltyScore"] = df.get("NoveltyScore", 0.0)

    # Extract all taxonomic ranks
    df = parse_all_ranks(df)

    # ------------------------------
    # NORMAL PHYLUM–GENUS DISTRIBUTION
    # ------------------------------
    for rank in ["Phylum", "Class", "Order", "Family", "Genus"]:
        counts = df[rank].value_counts()
        make_barplot(
            counts,
            f"{rank} Distribution (All Sequences)",
            f"{rank.lower()}_barplot_total.png",
            color="#2E7D32"
        )

    # ------------------------------
    # NOVELTY (>0.6) — deeper insight
    # ------------------------------
    novel_df = df[df["NoveltyScore"] > 0.6]

    for rank in ["Phylum", "Class", "Order", "Family", "Genus"]:
        counts = novel_df[rank].value_counts()
        make_barplot(
            counts,
            f"{rank} Distribution (Novel Sequences)",
            f"{rank.lower()}_barplot_novel.png",
            color="#C62828"
        )

    print("\n🎉 Biodiversity plots generated for all taxonomic levels!")
