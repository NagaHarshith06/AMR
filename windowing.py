import torch


class GenomeWindowGenerator:

    def __init__(self,window_size=512):
        self.window_size = window_size

    def create_windows(self,input_ids,attention_mask):
        windows = []
        cls_token = input_ids[0]
        start = 1

        while start < len(input_ids):
            end = start + self.window_size - 1
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]

            chunk_ids = torch.cat(
                [
                    cls_token.unsqueeze(0),
                    chunk_ids
                ]
            )

            chunk_mask = torch.cat(
                [
                    torch.tensor(
                        [1],
                        dtype=attention_mask.dtype
                    ),
                    chunk_mask
                ]
            )

            windows.append(
                (
                    chunk_ids,
                    chunk_mask
                )
            )
            start = end

        return windows