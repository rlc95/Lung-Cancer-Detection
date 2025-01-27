# from __future__ import division, print_function
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import glob
import re

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
#from flask_mail import Mail,Message

# Model saved with Keras model.save()
# MODEL_PATH = 'lungCancerDetection_model.h5'

# Load your trained model

#model = load_model(MODEL_PATH)

predict_dict = {0: 'Bengin cases', 1: 'Malignant cases', 2: 'Normal cases'}

# load json and create model
json_file = open('lungCancerDetection_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("lungCancerDetection_model.h5")
print("Loaded model from disk")

def model_predict(image,model2):
    img_width = 256
    img_height = 256
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #from keras.preprocessing import image
    from keras.utils import load_img, img_to_array


    #img = cv2.resize(img_save, (256, 256))
    # img = load_img(img_path, target_size=(256, 256))
    x = image.reshape(-1, 256, 256, 1)
    x = img_to_array(x)
    x = x / 255
    np.expand_dims(x, axis=0)
    preds = model2.predict(x)
    return preds

image = cv2.imread('Malignant.jpg')  # reads the image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV
# figure_size = 9  # the dimension of the x and y axis of the kernal.
cv2.imshow('image', image)
cv2.waitKey(0)

# The image will first be converted to grayscale
"""""
image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
cv2.imshow('HSV2BGR', image2)
cv2.waitKey(0)
"""


new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
figure_size = 9
cv2.imshow('grayscale', new_image)
cv2.waitKey(0)
plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image, cmap='gray'), plt.title('grayscale')
plt.xticks([]), plt.yticks([])
plt.show()

new_image_blur = cv2.blur(new_image, (figure_size, figure_size))
plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(new_image, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image_blur, cmap='gray'), plt.title('Mean filter')
plt.xticks([]), plt.yticks([])
plt.show()

new_image_median = cv2.medianBlur(new_image_blur, figure_size)
plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(new_image, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image_median, cmap='gray'), plt.title('Median Filter')
plt.xticks([]), plt.yticks([])
plt.show()

new_image_gauss = cv2.GaussianBlur(new_image_median, (figure_size, figure_size), 0)
plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(new_image, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image_gauss, cmap='gray'), plt.title('Gaussian Filter')
plt.xticks([]), plt.yticks([])
plt.show()

# image enhancement
hist, bins = np.histogram(new_image_gauss.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(new_image_gauss.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

img2 = cdf[new_image_gauss]
plt.imshow(img2, cmap='gray'), plt.title('Histogrm')
plt.show()

equ = cv2.equalizeHist(img2)
res = np.hstack((img2, equ))  # stacking images side-by-side
cv2.imwrite('res.png', res)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img2)
cv2.imwrite('clahe_2.jpg', cl1)

# Segmentation

# b,g,r = cv2.split(cl1)
# rgb_img = cv2.merge([r,g,b])
# gray = cv2.cvtColor(cl1,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image=img2, threshold1=100, threshold2=200)  # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.subplot(211), plt.imshow(image)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(thresh, 'gray')
plt.imsave(r'thresh.png', thresh)
plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

# noise removal
kernel = np.ones((2, 2), np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
# sure background area
sure_bg = cv2.dilate(closing, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
# Threshold
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]
plt.imsave(r'uploads/image.jpg', image)
cv2.imshow('markers', image)
cv2.waitKey(0)

# Save the file to ./uploads

"""""
basepath = os.path.dirname(__file__)
file_path = os.path.join(
basepath, 'uploads', secure_filename(img.filename))
img.save(file_path)
"""

image = cv2.imread("uploads/image.jpg")

# Make prediction
#preds = model_predict(image,model)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# x = np.array(X).reshape(-1, 256, 256, 1)
img_array = np.asarray(image)
x = img_array.reshape(-1, 256, 256, 1)

prediction = model.predict(x,verbose=1)
indices = prediction.argmax() #argmax() will return the largest value....
indices

#print(prediction)
#maxindex = int(np.argmax(prediction))
y_pred_bool = np.argmax(prediction , axis=1)
print(y_pred_bool)
# Print the predicted class (0 = no cancer, 1 = cancer)
if np.any(indices == 4):
    print("Malignant case ")
elif np.any(indices == 1):
    print("Normal case")
else:
    print("Bengin case")






