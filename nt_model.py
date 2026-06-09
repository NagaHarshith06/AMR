from transformers import AutoModelForMaskedLM

class NTModel:
    def __init__(self):
        self.model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            trust_remote_code=True
        )

    def get_hidden_size(self):
        return self.model.config.hidden_size

    def forward_window(self,input_ids,attention_mask):
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_hidden_states=True
        )

        return outputs.hidden_states[-1]

    def get_cls_embedding(self,input_ids,attention_mask):
        hidden_states = self.forward_window(
            input_ids,
            attention_mask
        )
        cls_embedding = hidden_states[:, 0, :]

        return cls_embedding