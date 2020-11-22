import numpy as np
import matplotlib.pyplot as plt

spectrograms = np.load('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/spectrograms_mel_normalized.npy')
numbers = np.load('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/numbers_one_hot_encoded.npy')

print(spectrograms.shape)
print(numbers.shape)
