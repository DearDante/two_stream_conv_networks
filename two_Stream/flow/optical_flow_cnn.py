import numpy as np
import h5py
import gc
import optical_flow_data as ofd
import pickle
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i+n]

def get_activations(model, layer, X_batch):
  pass

def getTrainData(chunk, nb_classes, img_rows, img_cols):
  X_train, Y_train = ofd.stackOF(chunk, img_rows, img_cols)
  if X_train.size > 0 and Y_train.size > 0:
    X_train /= 255
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
  return (X_train, Y_train)

def CNN():
  input_frames = 5
  batch_size = 8
  nb_classes = 2
  nb_epoch = 10
  img_rows, img_cols = 150, 150
  img_channels = 2*input_frames
  chunk_size = 64

  with open('../dataset/flow_train_data.pickle', 'rb') as f1:
    flow_train_data=pickle.load(f1)
  model = Sequential()
  
  model.add(Convolution2D(48, 7, 7, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Convolution2D(96, 5, 5, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  
  model.add(Convolution2D(256, 3, 3, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.7))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.8))

  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  gc.collect()
  
  sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.1)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)

  for e in range(nb_epoch):
    print('-'*40)
    print('Epoch', e)
    print('-'*40)
    instance_count = 0

    flag = 0
    keys = list(flow_train_data.keys())
    random.shuffle(keys)

    for chunk in chunks(keys, chunk_size):
      print(chunk)
      #if flag < 1:
      X_test, Y_test = getTrainData(chunk, nb_classes, img_rows, img_cols)
      #  flag += 1
      #  continue
      instance_count += chunk_size
      X_batch, Y_batch = getTrainData(chunk, nb_classes, img_rows, img_cols)
      print('X', X_batch)
      if X_batch.size >0 and Y_batch.size>0:
        loss = model.fit(X_batch, Y_batch, verbose=1, batch_size=batch_size, nb_epoch=1)
        if instance_count % 256 == 0:
          loss = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
          preds = model.predict(X_test)
          print(preds)
          print('-'*40)
          print(Y_test)
          comparisons = []
          maximum = np.argmax(Y_test, axis=1)
          for i, j in enumerate(maximum):
            comparisons.append(preds[i][j])
          with open('compare.txt', 'a') as f1:
            f1.write(str(comparisons))
            f1.write('\n\n')
          with open('loss.txt', 'a') as f2:
            f2.write(str(loss))
            f2.write('\n')
          model.save_weights('flow_stream_model.h5', overwrite=True)

if __name__ != "main":
  CNN()
