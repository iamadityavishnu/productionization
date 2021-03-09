import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import tensorboard

import torchvision

from image_classifier.model import ImageModel, transforms

LEARNING_RATE = 0.001
BATCH_SIZE = 32

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
        epoch_loss = 0
        for inp, label in train_loader:
            optimizer.zero_grad()
            op = model(inp)
            loss = criterion(op, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_value = epoch_loss / len(train_dataset)
        writer.add_scalars('Loss', {
            'Training': loss_value
        }, ep)
    writer.close()
    print("Done training")


def test():
    test_loss = 0
    with torch.no_grad():
        for inp, label in test_loader:
            op = model(inp)
            loss = criterion(op, label)
            test_loss += loss.item()
    print(f"Test loss: {test_loss / len(test_dataset)}")


if __name__ == '__main__':
    train(5)
    test()
