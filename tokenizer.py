from transformers import AutoTokenizer

class NTTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            trust_remote_code=True
        )

    def tokenize(self, sequence):
        encoded = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True
        )

        return encoded