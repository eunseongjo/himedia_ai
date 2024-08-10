import cv2
import numpy as np
from collections import Counter

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

import cv2

img = cv2.imread("cat_and_dog.jpg")
# cv2_imshow(img)
cv2.imshow("test", img)
cv2.waitKey(0)


# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("cat_and_dog.jpg")

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)

# DetectionResult(
#   detections=[
#     Detection(bounding_box=BoundingBox(origin_x=72, origin_y=162, width=252, height=191), 
#               categories=[Category(index=None, score=0.780297040939331, display_name=None, category_name='cat')], 
#               keypoints=[]), 
#     Detection(bounding_box=BoundingBox(origin_x=303, origin_y=27, width=249, height=345), 
#               categories=[Category(index=None, score=0.7625645399093628, display_name=None, category_name='dog')], 
#               keypoints=[])])

#STEP 5 : 찾은 객체의 종류와 갯수를 출력하시오
category_count = Counter()

for detection in detection_result.detections:
    for category in detection.categories:
        category_count[category.category_name] += 1

# 결과 출력
for category_name, count in category_count.items():
    print(f"{category_name}: {count}")


result_dict = {}
for detection in detection_result.detections:
   category = detection.categories[0].category_name
   if category not in result_dict:
      result_dict[category] = 1
   else:
      result_dict[category] += 1

print(result_dict)

# # STEP 5: Process the detection result. In this case, visualize it.
# image_copy = np.copy(image.numpy_view())
# annotated_image = visualize(image_copy, detection_result)
# rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# # cv2_imshow(rgb_annotated_image)
# cv2.imshow("test", rgb_annotated_image)
# cv2.waitKey(0)