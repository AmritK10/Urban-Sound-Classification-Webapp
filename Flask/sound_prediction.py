import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation
from tensorflow.keras.regularizers import l2
import numpy as np
import pickle

window_size = 507
frames = 128
bands = 128

# paper model translated
def build_model():
  #forming model
  model=Sequential()

  #adding layers and forming the model
  model.add(Conv2D(24,kernel_size=5,strides=1,padding="Same",input_shape=(bands, frames, 1)))
  model.add(MaxPooling2D(pool_size = (4,2)))
  model.add(Activation('relu'))

  model.add(Conv2D(48,kernel_size=5,strides=1,padding="same"))
  model.add(MaxPooling2D(pool_size = (4,2)))
  model.add(Activation('relu'))

  model.add(Conv2D(48,kernel_size=5,strides=1,padding="same"))
  model.add(Activation('relu'))

  model.add(Flatten())

  model.add(Dense(64,activity_regularizer=l2(0.001)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(10,activity_regularizer=l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Activation('softmax'))

  #compiling
  model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

  return model

new_model = build_model()
new_model.load_weights("Model_data/sound_model_weights.h5")

Class_Labels = {-1 : 'Audio file too short to make a prediction!', 0 : 'Air Conditioner', 1 : 'Car Horn', 
                2 : 'Children Playing', 3 : 'Dog Barking', 4 : 'Drilling', 5 : 'Engine Idling', 6 : 'Gunshot', 
                7 : 'Jackhammer', 8 : 'Siren', 9 : 'Street Music'}

def process_test_point(file_path):
  sound_clip, sr = librosa.load(file_path)
  if (sound_clip.shape[0] // window_size < 128):
    return None
  log_specgrams = []
  start = 0
  for no_frames in range(128):
    end = start + window_size
    if(len(sound_clip[start:end]) == int(window_size)):
        signal = sound_clip[start:end]
        melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
        logspec = librosa.amplitude_to_db(melspec)
        logspec = logspec.T.flatten()[:, np.newaxis].T
        log_specgrams.append(logspec)
    start = end
  log_specgrams = np.asarray(log_specgrams).reshape(frames,bands,1)
  return log_specgrams

def get_prediction(sound_path):
    test_point = process_test_point(sound_path)
    if test_point is None:
        return Class_Labels[-1]
    test_point = test_point.reshape(-1,128,128,1)
    pred_class = new_model.predict_classes(test_point)[0]
    return Class_Labels[pred_class]
