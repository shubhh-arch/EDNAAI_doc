import os

files = [
    "data/bacterial_16S.fasta",
    "data/bacterial_23S_reference.fasta",
    "data/fungii_18S_reference.fasta",
    "data/fungi_28S_reference.fasta"
    # Add more if you downloaded others
]

with open("data/combined_reference.fasta", "w") as outfile:
    for fname in files:
        with open(fname) as infile:
            outfile.write(infile.read())
            outfile.write("\n")

print("✅ Merged into data/combined_reference.fasta")
