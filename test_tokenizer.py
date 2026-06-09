from tokenizer import NTTokenizer

tokenizer = NTTokenizer()
sequence = "ATCGATCGATCGATCG"
tokens = tokenizer.tokenize(sequence)

print(tokens.keys())
print("Input IDs Shape:")
print(tokens["input_ids"].shape)
print("Attention Mask Shape:")
print(tokens["attention_mask"].shape)
print(tokens["input_ids"])