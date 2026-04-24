# CPE663 Major Assignment: Building a Mini Transformer and Producing a Small Benchmark

## Overview

This project implements a compact Transformer model from scratch for binary sequence classification on a synthetic dataset. The assignment focuses on understanding the core components of the Transformer architecture, including self-attention, positional encoding, and encoder blocks.

## Features

- **From-Scratch Implementation**: All Transformer components built without using pre-built modules like `torch.nn.Transformer`
- **Modular Design**: Clean separation of model components (embeddings, attention, feed-forward networks)
- **Benchmark Suite**: Comparative analysis of different model configurations
- **Synthetic Dataset**: Custom sequence classification task with configurable parameters

## Project Structure

- `model.py`: Transformer classifier implementation
- `data.py`: Dataset generation and preprocessing
- `train.py`: Training and evaluation scripts
- `benchmark.py`: Benchmarking different model variants
- `utils.py`: Utility functions for masking and parameter counting
- `report.md`: Detailed analysis and results

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib
   ```

## Usage

### Training a Model

```python
from train import train_model
from model import TransformerClassifier
from data import generate_dataset

# Generate dataset
train_data, val_data, test_data = generate_dataset()

# Create model
model = TransformerClassifier(
    vocab_size=5,  # A, B, C, D, PAD
    embed_dim=64,
    num_heads=4,
    num_layers=1,
    max_seq_len=20,
    ff_dim=128,
    dropout=0.1
)

# Train
train_losses, val_accs, train_time = train_model(model, train_loader, val_loader, epochs=10, lr=0.001)
```

### Running Benchmarks

```bash
python benchmark.py
```

## Model Variants

The project benchmarks four model configurations:

| Model | Positional Encoding | Heads | Layers | Parameters |
|-------|-------------------|-------|--------|------------|
| A     | Yes              | 1     | 1      | 33,985     |
| B     | Yes              | 4     | 1      | 33,985     |
| C     | No               | 4     | 1      | 33,985     |
| D     | Yes              | 4     | 2      | 67,457     |

## Results

Best performing model (Model D) achieves:
- Validation Accuracy: 97.7%
- Test Accuracy: 97.0%
- Training Time: 32.42 seconds

Key findings:
- Positional encoding is crucial for sequence understanding
- Multiple attention heads improve performance
- Additional layers provide marginal gains with increased complexity

## Author

AungkhinemoeLIM

## References

[1] Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).