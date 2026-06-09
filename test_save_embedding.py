import torch
from save_embedding import save_embedding

embedding = torch.randn(512)
save_embedding(genome_id="1352.1888",embedding=embedding)