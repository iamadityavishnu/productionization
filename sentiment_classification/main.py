import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import pipeline

from fastapi import FastAPI

app = FastAPI()
classifier = pipeline('sentiment-analysis')


@app.get('/classify/{text}')
def get_prediction(text: str):
    output = classifier(text)
    return output
