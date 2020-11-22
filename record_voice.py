import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

from scipy.io import wavfile
import glob
import os
import librosa
from skimage.transform import resize


sd.default.device = [10, 7]  # input (>), output (<)

volume = 1
fs = 44100
duration = 2

print("Starting to record...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=2)
sd.wait()

print("Playing back...")
sd.play(audio, fs)
sd.wait()

if audio.dtype != np.float32:
    assert audio.dtype == np.int16
    sound_np = np.divide(
        audio, 32768, dtype=np.float32
    )

model = load_model('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/model_autokeras.h5')

S = librosa.feature.melspectrogram(y=audio[:, 0], sr=fs, n_mels=128)
S_DB = librosa.power_to_db(S, ref=np.max)
S_DB_NORM = (S_DB - np.min(S_DB)) / (np.max(S_DB) - np.min(S_DB))
S_DB_NORM = resize(S_DB_NORM, (128, 128), 3)
S_DB_NORM = np.reshape(S_DB_NORM, (1, 128, 128, 1))

prediction = model.predict(S_DB_NORM)
result = np.argmax(prediction[0])
print("He predicho un:", result)
print(prediction)
print()
