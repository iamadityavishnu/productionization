import torch

from fastapi import FastAPI, File, UploadFile

from PIL import Image

from .model import ImageModel, transforms

PATH = './image_classifier/model_file/model.pth'

app = FastAPI()
model = ImageModel(predict=True)
model.load_state_dict(torch.load(PATH))


@app.post('/classify/')
def home(file: UploadFile = File(...)):
    img: Image.Image = Image.open(file.file)
    img = img.resize((32, 32))
    inp = transforms(img).unsqueeze(0)
    with torch.no_grad():
        op = model(inp)
    confidence = op.max().item()
    class_ = op.argmax(1).item()
    print(confidence, class_)
    return {
        'class': class_,
        'confidence': confidence
    }
