import cv2
from flask import Flask, redirect, url_for, request, render_template
server = Flask(__name__)

@server.route('/', methods=['GET'])
def index():
    return "<h1>Hello</h1>"

@server.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        return "<h1>Hello</h1>"

# server.run()

server.run(host="localhost", port=8080, debug=True)

import numpy as np
from keras.models import load_model
from PIL import Image

# Load the pre-trained ANN model
# model = load_model('D:\RLC\data Management\Project\pythonProject\lungCancerDetection_model.h5')
from keras.models import model_from_json

# load json and create model
json_file = open('lungCancerDetection_model4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("lungCancerDetection_model4.h5")
print("Loaded model from disk")

# Load the new CT scan image
img = Image.open('uploads/Bengin.jpg')

img_array = np.asarray(img)
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

"""""
from keras.utils import load_img, img_to_array

    #img = cv2.resize(img_save, (256, 256))
    # img = load_img(img_path, target_size=(256, 256))

img_array = np.asarray(img)
x = img_array.reshape(-1, 256, 256, 1)

#x = img.reshape(-1, 256, 256, 1)
# x = img_to_array(x)
x = x / 255
np.expand_dims(x, axis=0)

# Preprocess the image
# img = img.resize((256, 256))
# img = img.resize(-1, 256, 256, 1)
# img = img.resize((256, 256))
# img_array = np.array(img)
# img_array = img_array.reshape(1, -1)

#img = img.resize((256, 256), resample=Image.BILINEAR)
#img = np.array(img)
#img = np.expand_dims(img, axis=0)

# Make a prediction on the preprocessed image

prediction = model.predict(x)
print(prediction)
maxindex = int(np.argmax(prediction))
pred_class = np.argmax(prediction, axis=1)
print(maxindex)

# Map predicted class label to class name
class_names = ['Benign', 'Malignant', 'Normal']
predicted_class_name = class_names[pred_class[0]]

# Print predicted class label
print('Predicted class:', predicted_class_name)
print('result', pred_class)
# Print the predicted class (0 = no cancer, 1 = cancer)
if np.any(maxindex == 0):
    print("B.")
elif np.any(maxindex == 1):
    print("ma.")
else:
    print("normal")
"""""

