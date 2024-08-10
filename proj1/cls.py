# import cv2
# import math

# IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0)


# Preview the images

  # STEP 1: Import the necessary modules. 추론기 모듈 가져오기
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론 객체 만들기(모델정보 필요)
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite2.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

# STEP 3: Load the input image. 추론시킬 이미지 가져오기
image = mp.Image.create_from_file("cat.jpg")

# STEP 4: Classify the input image. 추론 시키기
classification_result = classifier.classify(image)
print(classification_result)

# STEP 5: Process the classification result. In this case, visualize it. 사용자에게 보여지는 출력방식
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")
