import torch

from preprocessing import preprocess_sequence
from tokenizer import NTTokenizer
from windowing import GenomeWindowGenerator
from nt_model import NTModel
from pooling import mean_pool


class EmbeddingGenerator:

    def __init__(self):

        self.tokenizer = NTTokenizer()

        self.window_generator = GenomeWindowGenerator(
            window_size=512
        )

        self.model = NTModel()

    def generate_embedding(
        self,
        sequence
    ):

        sequence = preprocess_sequence(
            sequence
        )

        encoded = self.tokenizer.tokenize(
            sequence
        )

        input_ids = encoded["input_ids"][0]

        attention_mask = encoded["attention_mask"][0]

        windows = self.window_generator.create_windows(
            input_ids,
            attention_mask
        )

        cls_embeddings = []
        with torch.no_grad():
            for input_ids_window, attention_mask_window in windows:

                cls_embedding = self.model.get_cls_embedding(
                    input_ids_window,
                    attention_mask_window
                )

                cls_embeddings.append(
                    cls_embedding.squeeze(0)
                )

        cls_embeddings = torch.stack(
            cls_embeddings
        )

        genome_embedding = mean_pool(
            cls_embeddings
        )

        return genome_embedding