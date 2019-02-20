from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

import h5py
size = 64


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size,size,3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# model.compile(optimizer='adam', loss='binary_crossentropy',
                   # metrics=['accuracy'])


train_imagedata  = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
      zoom_range = 0.2, horizontal_flip=True)
test_imagedata   = ImageDataGenerator(rescale=1. / 255)
training_set     = train_imagedata.flow_from_directory('data_set1/train', target_size=(size,size), batch_size=32, class_mode='categorical')
test_set         = test_imagedata.flow_from_directory('data_set1/test', target_size=(size,size), batch_size=32, class_mode='categorical')
history          = model.fit_generator(training_set, steps_per_epoch=5, epochs=8,validation_data=test_set,validation_steps=80)

model.save('model1.h5')


# size=128
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size,size,1)))
# model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))

# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer=optimizers.RMSprop(lr=0.0003), loss='categorical_crossentropy', metrics=['acc'])


# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1.255)

# train_generator = train_datagen.flow_from_directory('data_set1/train',target_size=(size,size),batch_size=64, class_mode='categorical')
# test_generator  = test_datagen.flow_from_directory('data_set1/test', target_size=(size,size), batch_size=64, class_mode='categorical')

# model.fit_generator(train_generator, epochs=20, steps_per_epoch=50, validation_data=test_generator, validation_steps=7)

# model.save('model.h5')