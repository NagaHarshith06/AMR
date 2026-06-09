import torch

def mean_pool(cls_embeddings):
    """
    cls_embeddings shape:(num_windows, 512)
    returns:(512,)
    """
    genome_embedding = torch.mean(
        cls_embeddings,
        dim=0
    )
    return genome_embedding