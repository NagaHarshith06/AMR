from fna_reader import read_fna_file
from embedding_generator import EmbeddingGenerator


generator = EmbeddingGenerator()

sequence = read_fna_file(
    "fna_files/1352.1888.fna"
)

embedding = generator.generate_embedding(
    sequence
)

print(
    "Embedding Shape:",
    embedding.shape
)

print(
    embedding[:10]
)