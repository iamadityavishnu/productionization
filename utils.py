import torch
from sklearn.metrics import recall_score, precision_score, f1_score


def accuracy(output: torch.Tensor, labels: torch.Tensor):
    shape = len(labels)
    return (output.argmax(1) == labels).to(int).sum() / shape


def precision(output, labels):
    return precision_score(
        labels.flatten(),
        output.argmax(1).flatten(),
        average='weighted',
        zero_division=0
    )


def recall(output, labels):
    return recall_score(
        labels.flatten(),
        output.argmax(1).flatten(),
        average='weighted'
    )


def f1(output, labels):
    return f1_score(
        labels.flatten(),
        output.argmax(1).flatten(),
        average='weighted'
    )
