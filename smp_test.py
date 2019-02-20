# import numpy as np
# from keras.preprocessing import image
# from keras.models import load_model
# from PIL import Image
# from keras import models
# from keras.preprocessing.image import ImageDataGenerator
# import cv2

# model = load_model('model.h5')

# test_image =image.load_img('data_set/test/sample6.jpg', target_size=(64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = model.predict(test_image)

# train_imagedata  = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
#       zoom_range = 0.2, horizontal_flip=True)
# test_imagedata   = ImageDataGenerator(rescale=1. / 255)
# training_set     = train_imagedata.flow_from_directory('data/train', target_size=(64, 64), batch_size=32, class_mode='binary')
# test_set         = test_imagedata.flow_from_directory('data/test', target_size=(64, 64), batch_size=32, class_mode='binary')

# for k,v in training_set.class_indices.items():
#     if v==int(result[0][0]):
#         print(k)
# print(int(result[0][0]))
# print(training_set.class_indices)

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import cv2

model = load_model('model1.h5')
size = 64

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
key=cv2.waitKey(1)
# # test_image = image.load_img('data_set/test/sample5.jpg', target_size = (size,size))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# #training_set.class_indices


train_imagedata  = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
      zoom_range = 0.2, horizontal_flip=True)
test_imagedata   = ImageDataGenerator(rescale=1. / 255)
training_set     = train_imagedata.flow_from_directory('data_set1/train', target_size=(size,size), batch_size=32, class_mode='categorical')
test_set         = test_imagedata.flow_from_directory('data_set1/test', target_size=(size,size), batch_size=32, class_mode='categorical')
for k,v in training_set.class_indices.items():
    if v==int(prediction[0][0]):
        print(k)
print(int(prediction[0][0]))
'''
if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'
print(prediction)'''