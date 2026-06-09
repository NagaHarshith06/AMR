import torch

from nt_model import NTModel

model = NTModel()

input_ids = torch.randint(
    low=0,
    high=100,
    size=(512,)
)

attention_mask = torch.ones(
    512,
    dtype=torch.long
)

cls_embedding = model.get_cls_embedding(
    input_ids,
    attention_mask
)

print(
    "CLS Shape:",
    cls_embedding.shape
)