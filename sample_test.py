import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start = time.time()

#Define Path
model_path = 'model2.h5'
# model_weights_path = './models/weights.h5'
test_path = 'data_set/test_data/Manikantha .138.jpg'

#Load the pre-trained models
model = load_model(model_path)
# model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 150, 150

#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  #print(result)
  answer = np.argmax(result)
  # if answer == 0:
  #   print("Predicted: Manasa")
  if answer == 0:
    print("Predicted: Manikantha")
  elif answer == 1:
    print("Predicted: Narendra")
  elif answer == 2:
    print("Predicted: Simha")
  elif answer == 3:
    print("Predicted: Sravani")
  elif answer == 4:
    print("Predicted: Sudha")  

  print(answer)

#Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    
    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(result)

#Calculate execution time
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")

