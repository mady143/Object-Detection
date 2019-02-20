
# import numpy as np
# from keras.preprocessing import image
# from keras.models import load_model
# from PIL import Image
# from keras import models
# from keras.preprocessing.image import ImageDataGenerator
# import cv2

# model = load_model('model.h5')
# size = 200
# test_image =image.load_img('/home/itm-it1018/Downloads/webcam-model-master/data_set/test-data/Manikantha .138.jpg', target_size=(size,size))
# # print(test_image)
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = model.predict(test_image)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
# validation_datagen = ImageDataGenerator(rescale=1.255)

# train_generator = train_datagen.flow_from_directory('data_set/train',target_size=(size,size),batch_size=32, class_mode='categorical')
# validation_generator = validation_datagen.flow_from_directory('data_set/test', target_size=(size,size), batch_size=32, class_mode='categorical')
# for k,v in train_generator.class_indices.items():
#     if v==int(result[0][0]):
#         print(k)
# print(int(result[0][0]))
# print(train_generator.class_indices)

























import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import cv2



model = load_model('model.h5')
size = 128

video = cv2.VideoCapture(0)
# while True:
_, frame = video.read()
#print("frame",frame)
#Convert the captured frame into RGB
im = Image.fromarray(frame, 'RGB')

#Resizing into 128x128 because we trained the model with this image size.
im = im.resize((size,size))
img_array = np.array(im)
#print("frame123",img_array)
#Our keras model used a 4D tensor, (images x height x width x channel)
#So changing dimension 128x128x3 into 1x128x128x3 
img_array = np.expand_dims(img_array, axis=0)

#Calling the predict method on model to predict 'me' on the image
# prediction = int(model.predict(img_array)[0][0])
prediction = model.predict(img_array)
# print(prediction)
#if prediction is 0, which means I am missing on the image, then show the frame in gray color.
# if prediction == 1:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("Capturing", frame)
key=cv2.waitKey(225)
# # test_image = image.load_img('data_set/test/sample5.jpg', target_size = (size,size))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# #training_set.class_indices


from keras.preprocessing.image import ImageDataGenerator
train_data_dir = 'data_set1/train'
validation_data_dir = 'data_set1/test'
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

# test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(size,size),
        batch_size=batch_size,
        class_mode='categorical')

# validation_generator = datagen.flow_from_directory(
#         validation_data_dir,
#         target_size=(size,size),
#         batch_size=batch_size,
#         class_mode='categorical')
for k,v in train_generator.class_indices.items():
    if v==int(prediction[0][0]):
        print(k)
