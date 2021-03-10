import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import tensorboard

import torchvision

from image_classifier.model import ImageModel, transforms
from utils import accuracy, precision, recall, f1


LEARNING_RATE = 0.001
BATCH_SIZE = 32
PATH = './model_file/model.pth'

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transforms
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

model = ImageModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)


def train(epochs):

    print("Train start")
    writer = tensorboard.SummaryWriter(
        log_dir='./log', comment='Train loop'
    )
    for ep in range(1, epochs + 1):
        epoch_loss, epoch_accuracy, epoch_precision = 0, 0, 0
        epoch_f1, idx = 0, 0
        for idx, (inp, label) in enumerate(train_loader):
            optimizer.zero_grad()
            op = model(inp)
            loss = criterion(op, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_accuracy += accuracy(op, label)
            epoch_precision += precision(op, label)
            epoch_f1 += f1(op, label)
        writer.add_scalars('Training', {
            'Accuracy': epoch_accuracy / idx,
            'Precision': epoch_precision / idx,
            'F1': epoch_f1 / idx
        }, ep)
        writer.add_scalars('Loss', {
            'Training': epoch_loss / idx
        }, ep)
    writer.close()
    torch.save(model.state_dict(), PATH)
    print("Done training")


def test():
    test_loss = 0
    test_accuracy = 0
    test_precision = 0
    test_f1 = 0
    idx = 0
    with torch.no_grad():
        for idx, (inp, label) in enumerate(test_loader):
            op = model(inp)
            loss = criterion(op, label)
            test_loss += loss.item()
            test_accuracy += accuracy(op, label)
            test_precision += precision(op, label)
            test_f1 += f1(op, label)
    print(f"Test loss: {test_loss / idx}")
    print(f"Test accuracy: {test_accuracy / idx}")
    print(f"Test precision: {test_precision / idx}")
    print(f"Test f1: {test_f1 / idx}")


if __name__ == '__main__':
    train(50)
    test()
