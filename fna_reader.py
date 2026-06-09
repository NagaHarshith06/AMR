def read_fna_file(filepath):
    sequence_parts = []

    with open(filepath,"r") as file:
        for line in file:
            if line.startswith(">"):
                continue

            sequence_parts.append(line.strip())

    return "".join(sequence_parts)