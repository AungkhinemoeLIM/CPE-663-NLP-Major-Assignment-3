import torch

def generate_padding_mask(sequences, pad_token_id=0):
    return (sequences != pad_token_id).float()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
