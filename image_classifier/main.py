import torch

from fastapi import FastAPI, File, UploadFile

from PIL import Image

from .model import ImageModel, transforms

PATH = './image_classifier/model_file/model.pth'
ALLOWED_FILE_TYPES = ('jpg', 'jpeg')

app = FastAPI()
model = ImageModel(predict=True)
model.load_state_dict(torch.load(PATH))


@app.post('/classify/')
def home(file: UploadFile = File(...)):
    content_type, file_ext = file.content_type.split('/')
    print(content_type)
    if content_type != 'image' and file_ext.lower() not in ALLOWED_FILE_TYPES:
        return {
            'message': 'Unsupported file type'
        }
    img: Image.Image = Image.open(file.file)
    img = img.resize((32, 32))
    inp = transforms(img).unsqueeze(0)
    with torch.no_grad():
        op = model(inp)
    confidence = op.max().item()
    class_ = op.argmax(1).item()
    return {
        'class': class_,
        'confidence': confidence
    }
