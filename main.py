# --- Import necessary libraries ---
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from diffusers import AutoPipelineForText2Image
import torch
from rembg import remove
from PIL import Image
import io
# --- Import Pydantic to define request body ---
from pydantic import BaseModel

# --- Define the shape of our request body ---
class ImageRequest(BaseModel):
    prompt: str

# --- App Setup ---
app = FastAPI()

# --- Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16

try:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch_dtype,
        variant="fp16" if device == "cuda" else "fp32"
    )
    pipe.to(device)
    print("AI model loaded successfully.")
except Exception as e:
    print(f"Error loading AI model: {e}")
    pipe = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "AI Engine is running"}

# --- THE FIX IS HERE ---
# We change the function signature to expect an `ImageRequest` object from the body.
@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    if not pipe:
        raise HTTPException(status_code=500, detail="AI model is not available.")
    try:
        # We now access the prompt via `request.prompt`
        image = pipe(prompt=request.prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        
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
