import cv2
import numpy as np
from scipy.io import wavfile
import glob
import os
import matplotlib.pyplot as plt
import librosa
from skimage.transform import resize


spectrograms = []
numbers = []

files = glob.glob('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/augmented/*.wav')
for file in files:
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

    spectrograms.append(S_DB_NORM)
    numbers.append(number_ohe)


spectrograms = np.array(spectrograms)
numbers = np.array(numbers)

np.save('spectrograms_mel_normalized.npy', spectrograms)
np.save('numbers_one_hot_encoded.npy', numbers)
