import torch
import random
import time
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

from data import generate_dataset, VOCAB_SIZE, MAX_SEQ_LEN, PAD
from model import TransformerClassifier
from train import train_model, evaluate_model
from utils import count_parameters

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")

    # Generate datasets once for all experiments
    print("Generating datasets...")
    train_data, train_labels = generate_dataset(num_samples=5000)
    val_data, val_labels = generate_dataset(num_samples=1000)
    test_data, test_labels = generate_dataset(num_samples=1000)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Hyperparameters common to all models
    EMBED_DIM = 64
    FF_DIM = 128
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10

    benchmark_results = []
    all_train_losses = {}
    all_val_accuracies = {}

    # Define model configurations for benchmarking
    model_configs = [
        {"alias": "Model A", "use_positional_encoding": True, "num_heads": 1, "num_layers": 1},
        {"alias": "Model B", "use_positional_encoding": True, "num_heads": 4, "num_layers": 1},
        {"alias": "Model C", "use_positional_encoding": False, "num_heads": 4, "num_layers": 1},
        {"alias": "Model D", "use_positional_encoding": True, "num_heads": 4, "num_layers": 2},
    ]

    for config in model_configs:
        alias = config["alias"]
        print(f"\n--- Running {alias} ---")
        torch.manual_seed(42)
        random.seed(42)

        model = TransformerClassifier(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=config["num_heads"],
            ff_dim=FF_DIM,
            num_layers=config["num_layers"],
            dropout_rate=DROPOUT_RATE,
            use_positional_encoding=config["use_positional_encoding"]
        )
        model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        param_count = count_parameters(model)
        train_losses, val_accuracies, training_time = train_model(
            model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device
        )

        all_train_losses[alias] = train_losses
        all_val_accuracies[alias] = val_accuracies

        val_acc = evaluate_model(model, val_loader, device)
        test_acc = evaluate_model(model, test_loader, device)

        benchmark_results.append({
            "Model": alias,
            "Positional Encoding": "Yes" if config["use_positional_encoding"] else "No",
            "Heads": config["num_heads"],
            "Layers": config["num_layers"],
            "Val Acc": f"{val_acc:.4f}",
            "Test Acc": f"{test_acc:.4f}",
            "Train Time (s)": f"{training_time:.2f}",
            "Parameters": param_count
        })

    df_results = pd.DataFrame(benchmark_results)
    print("\n--- Benchmark Results ---")
    print(df_results.to_markdown(index=False))

    with open("benchmark_results.md", "w") as f:
        f.write(df_results.to_markdown(index=False))

    # Plotting Training Curves
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    for alias, losses in all_train_losses.items():
        plt.plot(range(1, EPOCHS + 1), losses, label=alias)
    plt.title("Training Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    for alias, accs in all_val_accuracies.items():
        plt.plot(range(1, EPOCHS + 1), accs, label=alias)
    plt.title("Validation Accuracy by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Training curves saved to training_curves.png")

if __name__ == "__main__":
    run_benchmark()
