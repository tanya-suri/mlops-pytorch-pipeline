import io
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from dataset import get_transforms
from model import get_model

app = FastAPI(title="CIFAR-10 Inference Service")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/checkpoints/classifier_v1.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
transform = get_transforms(train=False)

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        print(f"Warning: Model checkpoint not found at {MODEL_PATH}")
        return
    try:
        model = get_model(architecture="resnet18", num_classes=10)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
            
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        prob_dict = {classes[i]: round(probabilities[i], 4) for i in range(len(classes))}
        
        return JSONResponse(content={
            "predicted_class": classes[int(torch.argmax(outputs))],
            "probabilities": prob_dict
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
