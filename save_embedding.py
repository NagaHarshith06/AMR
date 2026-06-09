import os
import pandas as pd

def save_embedding(genome_id, embedding):
    os.makedirs(
        "embeddings",
        exist_ok=True
    )

    embedding_np = (
        embedding
        .detach()
        .cpu()
        .numpy()
    )

    filepath = (
        f"embeddings/"
        f"{genome_id}_embedding.csv"
    )

    pd.DataFrame([embedding_np]).to_csv(
        filepath,
        index=False
    )

    print(f"Saved: {filepath}")
    print(f"Shape: {embedding.shape}")