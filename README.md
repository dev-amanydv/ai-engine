# AI Engine for Blendlab
This repository contains the backend microservice for a generative AI app. It's a simple, powerful server built with Python and FastAPI, designed to handle heavy AI tasks like text-to-image generation and background removal.

This engine is specifically designed to be deployed as a Docker container on Hugging Face Spaces to leverage their free "CPU upgrade" tier, which provides the 16GB of RAM necessary to run the AI models.

## Features
Text-to-Image Generation: Creates high-quality images from text prompts using Stable Diffusion.

Background Removal: Precisely removes the background from any uploaded image.

Optimized for Production: Loads AI models only once on startup for fast and efficient request handling.

## Tech Stack
Backend Framework: FastAPI

AI Models:

Text-to-Image: Stability AI's SDXL-Turbo

Background Removal: rembg (U2-Net)

Core Libraries: torch, diffusers, Pillow

Deployment: Docker, Hugging Face Spaces

## API Endpoints
The server exposes the following endpoints:

#POST `/generate-image`
Generates an image from a text prompt.

Request Body:

prompt (string, required): The text description of the image you want to create.

Example Usage (cURL):
```
curl -X POST "https://your-hf-space-url/generate-image" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A futuristic cityscape at sunset, cinematic lighting"}' \
     --output my_generated_image.png
```

Success Response:
```
Code: 200 OK
```

Content: The generated image file (image/png).

POST `/remove-background`
Removes the background from an uploaded image.

Request Body:

file (file, required): The image file to be processed.

Example Usage (cURL):
```
curl -X POST "https://your-hf-space-url/remove-background" \
     -F "file=@/path/to/your/image.jpg" \
     --output image_no_background.png
```
Success Response:
```
Code: 200 OK
```
Content: The processed image file with a transparent background (image/png).
