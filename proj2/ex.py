# STEP 1 : import module
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

#STEP2 : create inference module
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

#STEP3 : load import image
img1 = cv2.imread("bbg1.jpeg")
img2 = cv2.imread("lhu.jpeg")

#STEP4 : inference
faces1 = app.get(img1)
faces2 = app.get(img2)
print(len(faces1))
print(len(faces2))

# assert len(faces1)==1
# assert len(faces2)==1
# print(faces)

# # STEP 5: draw detection result
# rimg = app.draw_on(img, faces)
# # cv2.imshow("test", rimg)
# # cv2.waitKey(0)
# cv2.imwrite("./t1_output.jpg", rimg)

# STEP 5: face similarity 어떻게 응용할 것인지
# then print all-to-all face similarity
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat1, feat2.T)
print(sims)
