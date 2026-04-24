import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from data import generate_dataset, VOCAB_SIZE, MAX_SEQ_LEN, PAD
from model import TransformerClassifier
from utils import generate_padding_mask, count_parameters

def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    train_losses = []
    val_accuracies = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()

            mask = generate_padding_mask(sequences, PAD).to(device)
            outputs = model(sequences, mask=mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_acc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    return train_losses, val_accuracies, training_time

def evaluate_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            mask = generate_padding_mask(sequences, PAD).to(device)
            outputs = model(sequences, mask=mask)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    return correct_predictions / total_samples

if __name__ == "__main__":
    # Hyperparameters (example values)
    EMBED_DIM = 64
    FF_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 1
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    USE_POS_ENC = True

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate datasets
    train_data, train_labels = generate_dataset(num_samples=5000)
    val_data, val_labels = generate_dataset(num_samples=1000)
    test_data, test_labels = generate_dataset(num_samples=1000)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        use_positional_encoding=USE_POS_ENC
    )

    print(f"Model parameters: {count_parameters(model)}")

    # Train model
    train_losses, val_accuracies, training_time = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device
    )

    # Evaluate on test set
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
