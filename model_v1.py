from sklearn.utils import shuffle
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import set_session
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


spectrograms = np.load('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/spectrograms_mel_normalized.npy')
spectrograms = np.expand_dims(spectrograms, axis=-1)
numbers = np.load('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/numbers_one_hot_encoded.npy')
spectrograms, numbers = shuffle(spectrograms, numbers)

CONV_FILTERS = [8, 16]
BATCH_SIZE = 8
EPOCHS = 100

model = Sequential()

model.add(Conv2D(CONV_FILTERS[0], (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(CONV_FILTERS[1], (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.summary()

optimizer = RMSprop(lr=1e-6)
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=0.0001, verbose=1)
]
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.fit(x=spectrograms, y=numbers, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.15, callbacks=callbacks)

model.save('model_v1_2.h5')
