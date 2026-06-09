VALID_NUCLEOTIDES = {
    "A",
    "T",
    "C",
    "G",
    "N"
}

def clean_sequence(sequence):
    """
    Remove invalid characters and
    convert everything to uppercase.
    """
    sequence = sequence.upper()
    cleaned = []

    for char in sequence:
        if char in VALID_NUCLEOTIDES:
            cleaned.append(char)

    return "".join(cleaned)

def preprocess_sequence(sequence):
    """
    Complete preprocessing pipeline.
    """
    sequence = clean_sequence(sequence)
    return sequence