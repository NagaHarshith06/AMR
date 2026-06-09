import os
import pandas as pd
from amr_dataset import AMRDataset
from fna_reader import read_fna_file
from embedding_generator import (EmbeddingGenerator)
from save_embedding import (save_embedding)

def main():

    csv_path = "classification.csv"
    fna_folder = "fna_files"
    dataset = AMRDataset(csv_path,fna_folder)
    generator = EmbeddingGenerator()

    genome_ids = (
        dataset.data["Genome ID"]
        .dropna()
        .astype(str)
        .unique()
    )
    #genome_ids = genome_ids[:1]
    print(f"Found {len(genome_ids)} genomes")

    for genome_id in genome_ids:
        
        fna_path = os.path.join(fna_folder,f"{genome_id}.fna")

        if not os.path.exists(fna_path):
            print(f"Missing: {fna_path}")
            continue

        print(f"Processing {genome_id}")
        sequence = read_fna_file(fna_path)
        embedding = (generator.generate_embedding(sequence))
        save_embedding(genome_id,embedding)

    print("\nAll embeddings generated.")

if __name__ == "__main__":
    main()