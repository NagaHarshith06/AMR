from tokenizer import NTTokenizer
from windowing import GenomeWindowGenerator

tokenizer = NTTokenizer()
sequence = "ATCG" * 5000
encoded = tokenizer.tokenize(sequence)
input_ids = encoded["input_ids"][0]
attention_mask = encoded["attention_mask"][0]

window_generator = GenomeWindowGenerator(window_size=512)

windows = window_generator.create_windows(
    input_ids,
    attention_mask
)

print("Number of windows:",len(windows))
print("First Window Shape:",windows[0][0].shape)