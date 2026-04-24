import torch
import random

# Vocabulary
PAD = 0
A = 1
B = 2
C = 3
D = 4
VOCAB_SIZE = 5 # PAD, A, B, C, D

# Sequence parameters
MAX_SEQ_LEN = 20
MIN_TRUE_SEQ_LEN = 6
MAX_TRUE_SEQ_LEN = 20

def compute_label(seq):
    """
    seq: list of tokens WITHOUT padding
    """
    length = len(seq)
    mid = length // 2
    first_token = seq[0]
    second_half = seq[mid:]
    return 1 if first_token in second_half else 0

def generate_sequence():
    true_len = random.randint(MIN_TRUE_SEQ_LEN, MAX_TRUE_SEQ_LEN)
    # Generate tokens from A, B, C, D (1, 2, 3, 4)
    seq = [random.randint(A, D) for _ in range(true_len)]
    label = compute_label(seq)

    # Pad the sequence
    padded_seq = seq + [PAD] * (MAX_SEQ_LEN - true_len)
    return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def generate_dataset(num_samples):
    sequences = []
    labels = []
    for _ in range(num_samples):
        seq, label = generate_sequence()
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.stack(labels)

# Example usage (for testing)
if __name__ == "__main__":
    train_data, train_labels = generate_dataset(1000)
    val_data, val_labels = generate_dataset(200)
    test_data, test_labels = generate_dataset(200)

    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Sample sequence: {train_data[0]}")
    print(f"Sample label: {train_labels[0]}")

    # Verify compute_label with examples from the assignment
    # Example 1: [A, C, B, D, A, PAD, PAD]
    # Valid seq = [A, C, B, D, A]
    # first_token = A, second_half = [B, D, A] -> label = 1
    test_seq_1 = [A, C, B, D, A]
    print(f"Test seq 1: {test_seq_1}, Label: {compute_label(test_seq_1)}")

    # Example 2: [B, C, D, A, C, PAD, PAD]
    # Valid seq = [B, C, D, A, C]
    # first_token = B, second_half = [A, C] -> label = 0
    test_seq_2 = [B, C, D, A, C]
    print(f"Test seq 2: {test_seq_2}, Label: {compute_label(test_seq_2)}")

    # Example 3: [A, B, C, D, PAD, PAD]
    # Valid seq = [A, B, C, D]
    # first_token = A, second_half = [C, D] -> label = 0
    test_seq_3 = [A, B, C, D]
    print(f"Test seq 3: {test_seq_3}, Label: {compute_label(test_seq_3)}")
