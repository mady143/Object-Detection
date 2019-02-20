
from keras import models
from keras import layers
from keras.layers import Dense
from keras import optimizers
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
# import PIL.Image
# from keras.preprocessing.image import ImageDataGenerator
import h5py

size=256
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size,size,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
# model.add(Dense(len(class_id_index)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.add(Dense(output_dim=64, input_dim=input_dim))
# model.add(Activation("relu"))
# model.add(Dense(output_dim=10))
# model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(optimizer=optimizers.RMSprop(lr=0.0003), loss='binary_crossentropy', metrics=['acc'])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.255)

train_generator = train_datagen.flow_from_directory('data_set/train',target_size=(size,size),batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory('data_set/test', target_size=(size,size), batch_size=32, class_mode='categorical')

model.fit_generator(train_generator,steps_per_epoch = 35,epochs = 18,validation_data = validation_generator,validation_steps = 150)
# model.LOAD_TRUNCATED_IMAGES = True
# model.fit_generator(train_generator,steps_per_epoch = 44,epochs = 22)
model.save('model1.h5')




# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense

# import h5py

# # Initialising the CNN
# model = Sequential()
# # Step 1 - Convolution
# model.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), activation = 'relu'))
# # Step 2 - Pooling
# model.add(MaxPooling2D(pool_size = (2, 2))) 
# # Adding a second convolutional layer
# model.add(Conv2D(32, (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# # Step 3 - Flattening
# model.add(Flatten())
# # Step 4 - Full connection
# model.add(Dense(units = 200, activation = 'relu'))
# model.add(Dense(units = 3, activation = 'softmax'))
# # Compiling the CNN
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# # Part 2 - Fitting the CNN to the images
# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale = 1./255,
# shear_range = 0.2,
# zoom_range = 0.2,
# horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255)
# training_set = train_datagen.flow_from_directory('data_set/train',
# target_size = (200, 200),batch_size = 32,class_mode = 'categorical')
# test_set = test_datagen.flow_from_directory('data_set/test',target_size = (200, 200),batch_size = 32,class_mode = 'categorical')

# print('>>>',training_set.class_indices)
# model.fit_generator(training_set,steps_per_epoch = 20,epochs = 25,validation_data = test_set,validation_steps = 150)

# model.save(model.h5)







# size=256


# model = models.Sequential()

# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size,size,3)))
# model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(2, 2))

# model.add(layers.Flatten())

# # model.add(LSTM(256, input_shape=(3, 3),return_sequences=False))
# # model.add(Dense(22, activation='softmax'))

# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))


# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# #sparse_categorical_crossentropy
# print(model.summary())
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1.255)

# train_generator = train_datagen.flow_from_directory('data_set/train', target_size=(size,size), batch_size=64, class_mode='categorical')


# test_generator = test_datagen.flow_from_directory('data_set/test', target_size=(size,size),batch_size=64, class_mode='categorical')


# history = model.fit_generator(train_generator, samples_per_epoch = 350, epochs = 25, validation_data = test_generator, nb_val_samples = 200)

# # scores = model.evaluate(history, verbose=0)
# # print("Accuracy: %.2f%%" % (scores[1]*100))

# model.save('model.h5')


