from rembg import remove
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import io, color, filters
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
from pydantic import BaseModel
import uuid
import os
from os import path
import asyncio
import threading

input_path = 'samples/input.jpg'
output_path = 'samples/output.png'

counter = 0

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

back_path = path.abspath(path.join(__file__, "../")
                         ).replace("\\", "/") + '/images'

app.mount("/images", StaticFiles(directory=back_path), name="images")

@app.post("/load_picture")
async def get_pic(width:int, height:int, file: bytes = File()):
    with open("samples/work.jpg", 'wb') as i:
         i.write(file)
    image = io.imread('samples/work.jpg')
    resized_image = skimage.transform.resize(image,  (int(height), int(width)))
    skimage.io.imsave('samples/input.jpg', resized_image)

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

    # Load PNG image
    image = io.imread('samples/output.png')

    mask = np.sum(image[:, :, :3], axis=2) > 0

    image_m = io.imread("samples/output.png")
    for c in range(3):
            image_m[:, :, c] = np.where(mask == 1,
                                    image_m[:, :, c] *
                                    0,
                                    image_m[:, :, c])
            
    image_AS = Image.fromarray(image_m)
    image_AS.save("samples/TTTT.png")

    # Create a new white image of the same size
    white = Image.new("RGB", (image.shape[1], image.shape[0]), (255, 255, 255))

    # Paste the original image into the white image using the mask
    white.paste(Image.fromarray(image_m), mask=Image.fromarray(image_m))

    # Save the result as a new PNG image
    white.save("samples/white_image.png")

@app.get("/get_pic")
async def get_pic(text, steps=30, mask_blur=4,denoising_strength=0.51,cfg_scale=17, width=1080, height=720):

    steps = steps
    text = text
    URL = "https://f285-193-41-142-48.ngrok.io/"
    from base64 import b64decode, b64encode

    image = io.imread('samples/hall.jpg')
    resized_image = skimage.transform.resize(image,  (int(height), int(width)))
    skimage.io.imsave('samples/hall_v2.jpg', resized_image)
    resized_image = Image.open("samples/hall_v2.jpg")
    image_m = io.imread("samples/output.png")
    resized_image.paste(Image.fromarray(image_m), mask=Image.fromarray(image_m))
    resized_image.save("samples/input_v2.jpg")

    UU = []

    with open("samples/work.jpg", "rb") as f:
        img1 = f.read()
        UU.append(b64encode(img1).decode())


    with open("samples/white_image.png", "rb") as f:
        img1 = f.read()
        YY = b64encode(img1).decode()


    res = requests.post(URL+'/sdapi/v1/img2img', headers={"Content-Type": 'application/json'}, json={
            "init_images": UU,
            "resize_mode": 1,
            "mask_blur": mask_blur,
            "prompt": text,
            "mask": None,
            "negative_prompt": "ugly, peoples",
            "steps": steps,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 4,
            "inpainting_mask_invert": 0,
            "inpainting_fill": 1,
            "sd_model_checkpoint": "realisticVisionV20_v20-inpainting.safetensors [b19fee4ab1]",
            "denoising_strength": denoising_strength,
            "cfg_scale": cfg_scale,
            "inpainting_mask_weight": 1,
            "initial_noise_multiplier": 1,
            "restore_faces": True,
            "width": width,
            "height": height,
            "sampler_index": "Euler a",
            "img2img_color_correction": False,
            "img2img_fix_steps": False,
            "img2img_background_color": "#000000",
            "include_init_images": True}
    )
    with open("test_neural.jpg", "wb") as f:
        f.write(b64decode(res.json()["images"][0]))

    # resized_image = Image.open("test_neural.jpg")
    # resized_image.paste(Image.fromarray(image_m), mask=Image.fromarray(image_m))
    # resized_image.save("test_neural.jpg")

    return FileResponse("test_neural.jpg")

#Создай ручку для получения списка всех файлов в папке images
@app.get("/get_list")
async def get_list():
    #Получаем список файлов в папке images
    files = os.listdir(back_path)
    #Верни файлы в формате Json
    return JSONResponse(content=files)


async def create_pic(name, json):
    URL = "https://f285-193-41-142-48.ngrok.io"
    res = requests.post(URL+'/sdapi/v1/img2img', headers={"Content-Type": 'application/json'}, json=json)
    from base64 import b64decode, b64encode

    with open("test_neural.jpg", "wb") as f:
        f.write(b64decode(res.json()["images"][0]))

    #Сгенерировать UUID и сохранить картинку с этим именем

    with open(back_path+"/"+name+".jpg", "wb") as f:
        f.write(b64decode(res.json()["images"][0]))

def between_callback(name, json):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(create_pic(name, json))
    loop.close()


@app.post("/webimg")
async def get_web_img(text: str = "[txt2mask]hairs[/txt2mask]punk hairstyle", 
                      steps: int = 30, 
                      mask_blur: int = 4, 
                      denoising_strength: float = 0.51, 
                      cfg_scale: int = 17, 
                      width: int = 1080, 
                      height: int = 720, 
                      webImg: UploadFile = File(...)):

    from base64 import b64decode, b64encode
    UU = []
    with open("web.jpg", "wb") as f:
        f.write(await webImg.read())

    with open("web.jpg", "rb") as f:
        img1 = f.read()
        UU.append(b64encode(img1).decode())
    
    name = str(uuid.uuid4())
    json={
            "init_images": UU,
            "resize_mode": 1,
            "mask_blur": mask_blur,
            "prompt": text,
            "mask": None,
            "negative_prompt": "ugly, peoples",
            "steps": steps,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 4,
            "inpainting_mask_invert": 0,
            "inpainting_fill": 1,
            "sd_model_checkpoint": "realisticVisionV20_v20-inpainting.safetensors [b19fee4ab1]",
            "denoising_strength": denoising_strength,
            "cfg_scale": cfg_scale,
            "inpainting_mask_weight": 1,
            "initial_noise_multiplier": 1,
            "restore_faces": True,
            "width": width,
            "height": height,
            "sampler_index": "Euler a",
            "img2img_color_correction": False,
            "img2img_fix_steps": False,
            "img2img_background_color": "#000000",
            "include_init_images": True}
    _thread = threading.Thread(target=between_callback, args=(name,json))
    _thread.start()

    # resized_image = Image.open("test_neural.jpg")
    # resized_image.paste(Image.fromarray(image_m), mask=Image.fromarray(image_m))
    # resized_image.save("test_neural.jpg")

    return {"name": name+".jpg"}
    


