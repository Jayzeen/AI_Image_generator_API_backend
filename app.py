from PIL import Image
import requests
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

authToken = os.getenv('TOKEN')

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
HEADERS = {"Authorization": f"Bearer {authToken}"}


def query_huggingface(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.content


@app.get("/")
def home():
    return {"Welcome": "AI Image generator API"}


@app.get("/generate")
async def generate(prompt: str):
    try:
        image_description = prompt

        if not image_description:
            raise HTTPException(
                status_code=400, detail="No input image description provided")

        image_bytes = query_huggingface({"inputs": image_description})
        image = Image.open(BytesIO(image_bytes))

        # Encode the image as base64
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        
        # Save the image locally as a PNG file
        image_filename = "generated_image.png"
        image.save(image_filename, format="PNG")

        # Return the base64-encoded image in the response
        return Response(content=image_data, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

