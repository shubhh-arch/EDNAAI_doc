from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import random

def random_dna(length=100):
    return ''.join(random.choices("ACGT", k=length))

records = [SeqRecord(Seq(random_dna(100)), id=f"Read_{i+1}", description="") for i in range(20)]

with open("data/sample1.fasta", "w") as f:
    SeqIO.write(records, f, "fasta")

print("✅ sample1.fasta with 20 random sequences created!")
