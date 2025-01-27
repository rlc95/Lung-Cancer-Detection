# Import Packages
# %config Completer.use_jedi = False

import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import seaborn as sns
import random
import os
import imageio
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from keras.preprocessing.image import ImageDataGenerator
from plotly.subplots import make_subplots
from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, \
    f1_score  # plot_confusion_matrix
import itertools
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE

import tensorflow as tf


directory = r'D:/RLC/data Management/Dataset/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']

# Image Size Variations
size_data = {}
for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    temp_dict = {}
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        height, width, channels = imageio.v2.imread(filepath).shape
        if str(height) + ' x ' + str(width) in temp_dict:
            temp_dict[str(height) + ' x ' + str(width)] += 1
        else:
            temp_dict[str(height) + ' x ' + str(width)] = 1

    size_data[i] = temp_dict

size_data
print(size_data)


for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        print(i)
        img = cv2.imread(filepath, 0)
        plt.imshow(img)
        plt.show()
        break


# Image Preprocessing and Testing
img_size = 256
for i in categories:
    cnt, samples = 0, 3
    fig, ax = plt.subplots(samples, 3, figsize=(15, 15))
    fig.suptitle(i)

    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for curr_cnt, file in enumerate(os.listdir(path)):
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath, 0)

        img0 = cv2.resize(img, (img_size, img_size))

        img1 = cv2.GaussianBlur(img0, (5, 5), 0)

        ax[cnt, 0].imshow(img)
        ax[cnt, 1].imshow(img0)
        ax[cnt, 2].imshow(img1)
        cnt += 1
        if cnt == samples:
            break

plt.show()


# Preparing Data
data = []
img_size = 256

for i in categories:
    path = os.path.join(directory, i)
    class_num = categories.index(i)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath, 0)
        # preprocess here
        img = cv2.resize(img, (img_size, img_size))
        data.append([img, class_num])

random.shuffle(data)

X, y = [], []
for feature, label in data:
    X.append(feature)
    y.append(label)

print('X length:', len(X))
print('y counts:', Counter(y))

# normalize
X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X / 255.0
y = np.array(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=10, stratify=y)

print(len(X_train), X_train.shape)
print(len(X_valid), X_valid.shape)

# Applying SMOTE to oversample the data
print(Counter(y_train), Counter(y_valid))

print(len(X_train), X_train.shape)

X_train = X_train.reshape(X_train.shape[0], img_size*img_size*1)

print(len(X_train), X_train.shape)

print('Before SMOTE:', Counter(y_train))
smote = SMOTE()
X_train_sampled, y_train_sampled = smote.fit_resample(X_train, y_train)
print('After SMOTE:', Counter(y_train_sampled))

X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
X_train_sampled = X_train_sampled.reshape(X_train_sampled.shape[0], img_size, img_size, 1)

print(len(X_train), X_train.shape)
print(len(X_train_sampled), X_train_sampled.shape)

'''''
classifier = Sequential()
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(6, kernel_initializer='he_uniform', activation='relu', input_shape= X_train.shape[1:]))

# classifier.add(Dense(units=6, init='he_uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
'''


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap = plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")

    else:
        print("Confusion Matrix without normalized")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment = 'center',
            color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


print("ANN - Model 1 with SMOTE data")

model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape= X_train.shape[1:]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax'),

])

# save model structure in jason file
model_json = model.to_json()
with open("lungCancerDetection_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
model.save_weights('lungCancerDetection_model.h5')


model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_sampled, y_train_sampled, batch_size=32, epochs=10, validation_data=(X_valid, y_valid))

model.evaluate(X_valid, y_valid)

y_pred = model.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_valid, y_pred_bool))
print('confusion_matrix', confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))
print('accuracy score', accuracy_score(y_true=y_valid, y_pred=y_pred_bool))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy - ANN with SMOTE data')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss - ANN with SMOTE data')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

y_pred_test = model.predict(X_valid).argmax(axis=1)
cm = confusion_matrix(y_valid, y_pred_test)
plot_confusion_matrix(cm, list(range(3)))

# misclassified_idx = np.where(y_pred_test == y_valid)[0]
# print('misclassified_idx',misclassified_idx)
# i = np.random.choice(misclassified_idx)
# cv2.imshow('image', X_valid[i])
# cv2.waitKey(0)
# plt.imshow(X_valid[i], cmap='gray')
# plt.title("true label : %s  predicted: %s" % (y_valid[i], y_pred_test[i]))


# Model Building with Class Weighted Approach
print("ANN - Model 2 with Class Weighted Approach")

model2 = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape= X_train.shape[1:]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax'),

])


# save model structure in jason file
model2_json = model2.to_json()
with open("lungCancerDetection_model2.json", "w") as json_file:
    json_file.write(model2_json)

# save trained model weight in .h5 file
model2.save_weights('lungCancerDetection_model2.h5')


model2.summary()

model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

new_weights = {
    0: X_train.shape[0]/(3*Counter(y_train)[0]),
    1: X_train.shape[0]/(3*Counter(y_train)[1]),
    2: X_train.shape[0]/(3*Counter(y_train)[2]),
}

# new_weights[0] = 0.5
# new_weights[1] = 20

# Define the true labels and predicted labels
y_true = np.array([0, 1, 2, 3])
y_pred = np.array([0, 0, 1, 1])

# Calculate precision and F-score with default settings (will raise warning)
precision = precision_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

# Calculate precision and F-score with zero_division parameter set to 1
precision = precision_score(y_true, y_pred, average=None, zero_division=1)
f1 = f1_score(y_true, y_pred, average=None, zero_division=1)

print("Precision:", precision)
print("F1-score:", f1)

new_weights

history = model2.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_valid, y_valid), class_weight=new_weights)

model2.evaluate(X_valid, y_valid)

y_pred = model2.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_valid, y_pred_bool))
print('confusion_matrix', confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))
print('accuracy score', accuracy_score(y_true=y_valid, y_pred=y_pred_bool))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy - ANN with Class Weighted Approach')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss - ANN with Class Weighted Approach')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

y_pred_test = model2.predict(X_valid).argmax(axis=1)
cm = confusion_matrix(y_valid, y_pred_test)
plot_confusion_matrix(cm, list(range(3)))


print("ANN - Model 5 with Data Augmentation")

train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train,  batch_size=32)
val_generator = val_datagen.flow(X_valid, y_valid,  batch_size=32)

model5 = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape= X_train.shape[1:]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax'),

])

# save model structure in jason file
model5_json = model5.to_json()
with open("lungCancerDetection_model5.json", "w") as json_file:
    json_file.write(model2_json)

# save trained model weight in .h5 file
model5.save_weights('lungCancerDetection_model5.h5')

model5.summary()

model5.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model5.fit_generator(train_generator, epochs=10, validation_data=val_generator, class_weight=new_weights)

y_pred = model5.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_valid, y_pred_bool))
print('confusion_matrix', confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))
print('accuracy score', accuracy_score(y_true=y_valid, y_pred=y_pred_bool))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy - ANN with Data Augmentation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss - ANN with Data Augmentation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

y_pred_test = model5.predict(X_valid).argmax(axis=1)
cm = confusion_matrix(y_valid, y_pred_test)
plot_confusion_matrix(cm, list(range(3)))



print("CNN - Model 3 with Class Weighted Approach")

model3 = Sequential()

model3.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Flatten())
model3.add(Dense(16))
model3.add(Dense(3, activation='softmax'))

model3.summary()

# save model structure in jason file
model3_json = model3.to_json()
with open("lungCancerDetection_model3.json", "w") as json_file:
    json_file.write(model3_json)

# save trained model weight in .h5 file
model3.save_weights('lungCancerDetection_model3.h5')

new_weights2 = {
    0: X_train.shape[0]/(3*Counter(y_train)[0]),
    1: X_train.shape[0]/(3*Counter(y_train)[1]),
    2: X_train.shape[0]/(3*Counter(y_train)[2]),
}

# new_weights[0] = 0.5
# new_weights[1] = 20

new_weights2

history = model2.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_valid, y_valid), class_weight= new_weights2)

y_pred = model3.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_valid, y_pred_bool))
print('confusion_matrix', confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))
print('accuracy score', accuracy_score(y_true=y_valid, y_pred=y_pred_bool))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy - CNN with Class Weighted Approach')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss - CNN with Class Weighted Approach')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

y_pred_test = model3.predict(X_valid).argmax(axis=1)
cm = confusion_matrix(y_valid, y_pred_test)
plot_confusion_matrix(cm, list(range(3)))




print("CNN - Model 4 with SMOTE data")

model4 = Sequential()

model4.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
model4.add(Activation('relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))

model4.add(Conv2D(64, (3, 3), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))

model4.add(Flatten())
model4.add(Dense(16))
model4.add(Dense(3, activation='softmax'))

model4.summary()

# save model structure in jason file
model4_json = model4.to_json()
with open("lungCancerDetection_model4.json", "w") as json_file:
    json_file.write(model4_json)

# save trained model weight in .h5 file
model4.save_weights('lungCancerDetection_model4.h5')

model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model4.fit(X_train_sampled, y_train_sampled, batch_size=32, epochs=10, validation_data=(X_valid, y_valid))

y_pred = model4.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_valid, y_pred_bool))
print('confusion_matrix', confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))
print('accuracy score', accuracy_score(y_true=y_valid, y_pred=y_pred_bool))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy - CNN with SMOTE data')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss - CNN with SMOTE data')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

y_pred_test = model4.predict(X_valid).argmax(axis=1)
cm = confusion_matrix(y_valid, y_pred_test)
plot_confusion_matrix(cm, list(range(3)))
