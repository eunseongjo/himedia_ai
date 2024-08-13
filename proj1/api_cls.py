# STEP 1: Import the necessary modules. 추론기 모듈 가져오기
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
from fastapi import FastAPI, File, UploadFile

# STEP 2: Create an ImageClassifier object. 추론 객체 만들기(모델정보 필요)
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite2.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options) #추론기를 만들어 놓고 서버 호출을 진행해야 효율적임

app = FastAPI()

from PIL import Image
import io
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # STEP 3: Load the input image. 추론시킬 이미지 가져오기
    # 1. create pil image from http file
    # 1-1. convert http file to file
    pil_img = Image.open(io.BytesIO(contents))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image. 추론 시키기
    classification_result = classifier.classify(image)

    # STEP 5: Process the classification result. In this case, visualize it. 사용자에게 보여지는 출력방식
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"

    return {"result": result}

