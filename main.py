from fastapi import FastAPI
from pydantic import BaseModel

from RealESRGAN.inference_realesrgan import main as upscaleImage

class Item(BaseModel):
    name: str
    
app = FastAPI()

@app.get("/upscale")
async def test_funct():
	image = upscaleImage()
	
	return 1
