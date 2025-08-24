# --- Import necessary libraries ---
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from diffusers import AutoPipelineForText2Image
import torch
from rembg import remove
from PIL import Image
import io
from pydantic import BaseModel
import os

# --- Hugging Face Authentication ---
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# --- Define the shape of our request body ---
class ImageRequest(BaseModel):
    prompt: str

# --- App Setup ---
app = FastAPI()

# --- Model Loading ---
# Optimized device detection for Mac (MPS), CUDA, and CPU
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

print(f"✅ Using device: {device}")

pipe = None
try:
    print("Attempting to load AI model...")

    # --- UPDATED TO A MUCH FASTER TURBO MODEL ---
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", # Changed to the faster Turbo model
        torch_dtype=torch_dtype
    )
    pipe.to(device)

    print("AI model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading AI model: {e}")
    pipe = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "AI Engine is running"}

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    if not pipe:
        raise HTTPException(status_code=500, detail="AI model is not available. It may have failed to load on startup.")
    try:
        # --- UPDATED FOR THE TURBO MODEL ---
        # Turbo models need very few steps (1-4) and no guidance scale
        image = pipe(
            prompt=request.prompt,
            num_inference_steps=1,
            guidance_scale=0.0
        ).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {e}")

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    input_image_bytes = await file.read()
    try:
        output_image_bytes = remove(input_image_bytes)
        return StreamingResponse(io.BytesIO(output_image_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove background: {e}")