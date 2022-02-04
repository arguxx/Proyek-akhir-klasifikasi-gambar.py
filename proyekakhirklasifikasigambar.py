# -*- coding: utf-8 -*-
"""ProyekAkhirKlasifikasiGambar.ipynb

Proyek Akhir ML Klasifikasi Gambar
"""

# import lib yang dibutuhkan
import zipfile
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Activation,Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from google.colab import files

# Download zip file dari dicoding academy
!wget --no-check-certificate https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip

# Unzip file dan simpah di direktori tmp

local_zip = '/tmp/rockpaperscissors.zip'
# extract ke dalam direktori rps
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/rps')
zip_ref.close()

# menghapus file yang tidak dibutuhkan
os.remove("/rps/rockpaperscissors/README_rpc-cv-images.txt")
os.remove("/rps/rockpaperscissors/rps-cv-images/README_rpc-cv-images.txt")
shutil.rmtree('/rps/__MACOSX')

# mengecek kesiapan file
dir = '/rps'
print(os.listdir(dir))
dir = '/rps/rockpaperscissors'
print(os.listdir(dir))

# memindahkan data ke dalam list dan memberikan label
dataset=[]
mapping={"paper":0,"rock":1,"scissors":2}
count=0

for file in os.listdir(dir):
    if file=='README_rpc-cv-images.txt' or file=='rps-cv-images':
        continue
    path=os.path.join(dir,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im),target_size=(150,150))
        image=img_to_array(image)
        image=image/255.0
        dataset.append([image,count])
    count=count+1
data,labels = zip(*dataset)
print("Jumlah data: {}".format(len(data)))
print("Jumlah label: {}".format(len(labels)))

# buat gambar jadi array
labels=to_categorical(labels)
data=np.array(data)
labels=np.array(labels)
print("Data Shape:{}\nLabels shape: {}".format(data.shape,labels.shape))
print(len(labels))
# print(labels)

# melakukan preprocessing dan augmentasi gambar 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(#horizontal_flip=True,
                             #vertical_flip=True,
                             rotation_range=20,
                             zoom_range=0.2,
                             shear_range=0.2,
                             fill_mode="nearest"
                             )

# membuat test split
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(data,labels,test_size=0.4,random_state=44)

# buat model
model = Sequential()
# masukan configurasi
model.add(Conv2D(32, (3,3),input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
# layer 1
model.add(Dense(units=512, activation='relu'))
# layer 2
model.add(Dense(units=3, activation='softmax'))

model.summary()
# keras.Sequential()

# compile modelnya
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# menggunakan metode callback, untuk mendapatkan hasil yang optimal
my_calls = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=5),
            keras.callbacks.ModelCheckpoint("savedModel",verbose=1,save_best_only=True)]

# latih model tersebut
history=model.fit(
    train_datagen.flow(trainx,trainy,batch_size=32),
    validation_data=(testx,testy),
    callbacks = my_calls,
    epochs=20)

# buat result dari prediksi tersebut
y_pred=model.predict(testx)
pred=np.argmax(y_pred,axis=1)
ground = np.argmax(testy,axis=1)

print(classification_report(ground,pred))

# membuat grafik akurasi dan grafik loss
get_acc = history.history['accuracy']
value_acc = history.history['val_accuracy']
get_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs = range(len(get_acc))
plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('akurasi')
plt.legend(loc=0)
plt.figure()
plt.show()

epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='Accuracy of Training data')
plt.plot(epochs, validation_loss, 'b', label='Accuracy of Validation data')
plt.title('loss')
plt.legend(loc=0)
plt.figure()
plt.show()

# Commented out IPython magic to ensure Python compatibility.
from keras.preprocessing import image
# %matplotlib inline

# upload file
uploaded = files.upload()
 
# Melakukan prediksi gambar
for fn in uploaded.keys():
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  result = np.array_str(classes) 
  
  if result == paper :
     result = 'Kertas(Paper)'
  elif result == rock :
     result = 'Rock (Batu)'
  elif result == scissors :
     result = 'Gunting(Scissors)'

  print("ini adalah gambar: {} \nGambar diprediksi sebagai gambar: {}".format(fn, result))