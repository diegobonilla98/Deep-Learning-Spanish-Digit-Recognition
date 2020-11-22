from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

import numpy as np
from scipy.io import wavfile
import glob
import os
import matplotlib.pyplot as plt
import librosa
from skimage.transform import resize


model = load_model('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/model_autokeras.h5')

file = './test_audios/9_test_B.wav'

sample_rate, sound_np = wavfile.read(file)
if sound_np.dtype != np.float32:
    assert sound_np.dtype == np.int16
    sound_np = np.divide(
        sound_np, 32768, dtype=np.float32
    )
number = int(os.path.split(file)[-1][0])
number_ohe = np.eye(10)[number]

S = librosa.feature.melspectrogram(y=sound_np, sr=sample_rate, n_mels=128)
S_DB = librosa.power_to_db(S, ref=np.max)
S_DB_NORM = (S_DB - np.min(S_DB)) / (np.max(S_DB) - np.min(S_DB))
S_DB_NORM = resize(S_DB_NORM, (128, 128), 3)
S_DB_NORM = np.reshape(S_DB_NORM, (1, 128, 128, 1))

prediction = model.predict(S_DB_NORM)
result = np.argmax(prediction[0])
print(result)
print(prediction)
print()
