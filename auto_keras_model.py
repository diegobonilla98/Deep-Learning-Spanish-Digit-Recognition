import autokeras as ak
import numpy as np
from sklearn.utils import shuffle


spectrograms = np.load('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/spectrograms_mel_normalized.npy')
spectrograms = np.expand_dims(spectrograms, axis=-1)
numbers = np.load('/media/bonilla/HDD_2TB_basura/databases/Digitos_Spanish_Mios/numbers_one_hot_encoded.npy')
spectrograms, numbers = shuffle(spectrograms, numbers)

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=5)
clf.fit(spectrograms, numbers, validation_split=0.20, epochs=20)

model = clf.export_model()
model.save("model_autokeras.h5")
model.save("model_autokeras", save_format="tf")
