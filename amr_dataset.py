import os
import pandas as pd
from torch.utils.data import Dataset
#----------------------------------section 1
class AMRDataset(Dataset):

    def __init__(self, csv_file, fna_folder):
        self.data = pd.read_csv(csv_file)
        self.fna_folder = fna_folder

    def __len__(self):
        return len(self.data)

    def read_fna(self, filepath):
        sequence = []

        with open(filepath, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue
                sequence.append(line.strip())

        return "".join(sequence)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        genome_id = str(row["Genome ID"])
        antibiotic = row["Antibiotic"]
        label = int(row["Resistant Phenotype"])

        file_path = os.path.join(
            self.fna_folder,
            f"{genome_id}.fna"
        )

        sequence = self.read_fna(file_path)

        return {
            "genome_id": genome_id,
            "sequence": sequence,
            "antibiotic": antibiotic,
            "label": label
        }