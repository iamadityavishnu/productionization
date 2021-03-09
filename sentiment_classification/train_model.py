import random

import torch
import torch.optim as optim
from torchtext import data, datasets

from sentiment_classification.model import RNN

SEED = 0

torch.manual_seed(SEED)

tokenizer = data.utils.get_tokenizer('spacy')
train_iter, test_iter = datasets.IMDB(split=('train', 'test'))

