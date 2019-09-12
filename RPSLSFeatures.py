from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
print("TensorFlow version is ", tf.__version__)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#------------------------------------------------------
# The generators
#------------------------------------------------------


TRAINING_DIR = "/home/adrian/Desktop/RockScissorsPaperLizardSpock/DataSet2/Out/train"
TEST_DIR = "/home/adrian/Desktop/RockScissorsPaperLizardSpock/DataSet2/Out/test"


image_size = 96
batch_size = 64
validation_split = 0.20


def change_range(image):
  image /= 255.0  # normalize to [0,1] range
  return 2 * image - 1


training_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=change_range,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=change_range)


train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(image_size, image_size),
    class_mode='categorical',
    # subset='training',
    batch_size=batch_size,
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(image_size, image_size),
    class_mode='categorical',
    # subset='validation',
    batch_size=batch_size,
    shuffle=True
)

#------------------------------------------------------
# The model
#------------------------------------------------------


IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               alpha=0.35,
                                               include_top=False,
                                               weights='imagenet')  # random initialization

base_model.trainable = False


feature_model = tf.keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(5, activation='softmax')
])


optimizer = tf.keras.optimizers.RMSprop(lr=2e-5)
feature_model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=["accuracy"])


print(feature_model.summary())

epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size


history = feature_model.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      # workers=4,
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps)

feature_model.save("rpslsFeatures.h5")


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()
