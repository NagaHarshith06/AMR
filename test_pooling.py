import torch
from pooling import mean_pool

cls_embeddings = torch.randn(5,512)
genome_embedding = mean_pool(cls_embeddings)
print("Input Shape:",cls_embeddings.shape)
print("Output Shape:",genome_embedding.shape)