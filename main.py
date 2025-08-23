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
# Make sure you set your token like this:
# export HUGGINGFACE_TOKEN="your_token_here"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# --- Define the shape of our request body ---
class ImageRequest(BaseModel):
    prompt: str

# --- App Setup ---
app = FastAPI()

# --- Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

pipe = None
try:
    print("Attempting to load AI model...")

    pipe = AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4", # <-- CHANGED to a much smaller model
        torch_dtype=torch_dtype
        # Removed variant and token args for simplicity, auth is automatic
    )
    pipe.to(device)

    print("AI model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading AI model: {e}")
    print("👉 Make sure you have run: huggingface-cli login OR set HUGGINGFACE_TOKEN env variable.")
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
        image = pipe(
            prompt=request.prompt,
            num_inference_steps=20,
            guidance_scale=7.5
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