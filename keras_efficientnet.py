

# Importing the libraries
import os
import glob
import shutil
import json
import keras
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import albumentations


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

os.environ['TF_XLA_FLAGS']='--tf_xla_enable_xla_devices'
# Defining the directories
work_dir = '../input/cassava-leaf-disease-classification/'
train_path = '/kaggle/input/cassava-leaf-disease-classification/train_images'

work_dir = '../kaggle-cassava-classification/data/'
os.listdir(work_dir)
train_path = './data/train_images/'

tf.test.gpu_device_name()
data = pd.read_csv(work_dir + 'train.csv')
f = open(work_dir + 'label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k):v for k,v in real_labels.items()}

# Defining the working dataset
data['class_name'] = data.label.map(real_labels)

#data=data.head(1000)
train, val = train_test_split(data, test_size=0.1, random_state=42, stratify=data['class_name'])

IMG_SIZE = 380
size = (IMG_SIZE, IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 20

datagen_train = ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

datagen_val = ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                    )

train_data = datagen_train.flow_from_dataframe(train,
                             directory = train_path,
                             seed=42,
                             x_col = 'image_id',
                             y_col = 'class_name',
                             target_size = size,
                             #color_mode="rgb",
                             class_mode = 'categorical',
                             interpolation = 'nearest',
                             shuffle = True,
                             batch_size = BATCH_SIZE)

val_data = datagen_val.flow_from_dataframe(val,
                             directory = train_path,
                             seed=42,
                             x_col = 'image_id',
                             y_col = 'class_name',
                             target_size = size,
                             #color_mode="rgb",
                             class_mode = 'categorical',
                             interpolation = 'nearest',
                             shuffle = True,
                             batch_size = BATCH_SIZE)


def create_model():
    model = Sequential()
    model.add(ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation='softmax'))

    return model


leaf_model = create_model()
leaf_model.summary()

EPOCHS = 25
SST = train_data.n//train_data.batch_size
SSV = val_data.n//val_data.batch_size


def model_fitter():
    leaf_model = create_model()

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0001,
                                                   name='categorical_crossentropy')

    leaf_model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=['categorical_accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True, verbose=1)

    checkpoint_cb = ModelCheckpoint("Cassava_best_model.h5", save_best_only=True, monitor='val_loss', mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, mode='min', verbose=1)

    history = leaf_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             steps_per_epoch=SST,
                             validation_steps=SSV,
                             callbacks=[es, checkpoint_cb, reduce_lr])

    leaf_model.save('Cassava_model' + '.h5')

    return history

results = model_fitter()    