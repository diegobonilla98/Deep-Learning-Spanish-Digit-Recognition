from audiomentations import (
    Compose,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    AddGaussianSNR,
    Resample,
    ClippingDistortion,
)
import os
import numpy as np
from scipy.io import wavfile
import glob

import matplotlib.pyplot as plt


# now in different location
files = glob.glob('./dataset/*.wav')
for audio_file in files:
    sample_rate, sound_np = wavfile.read(audio_file)
    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        sound_np = np.divide(
            sound_np, 32768, dtype=np.float32
        )
    number = os.path.split(audio_file)[-1][:-4]

    transforms = [
        {"instance": AddGaussianSNR(p=1.0), "num_runs": 3},
        {"instance": TimeStretch(min_rate=0.4, max_rate=1.25, p=1.0), "num_runs": 5},
        {
            "instance": PitchShift(min_semitones=-5, max_semitones=5, p=1.0),
            "num_runs": 6,
        },
        {"instance": Shift(min_fraction=-0.85, max_fraction=0.85, p=1.0), "num_runs": 4},
        {"instance": Resample(p=1.0), "num_runs": 5},
        {"instance": ClippingDistortion(p=1.0), "num_runs": 3},
    ]

    for transform in transforms:
        augmenter = Compose([transform["instance"]])
        run_name = (
            transform.get("name")
            if transform.get("name")
            else transform["instance"].__class__.__name__
        )
        for i in range(transform["num_runs"]):
            output_file_path = os.path.join(
                'augmented', "{}_{}_{:03d}.wav".format(number, run_name, i)
            )
            augmented_samples = augmenter(samples=sound_np, sample_rate=sample_rate)
            wavfile.write(output_file_path, rate=sample_rate, data=augmented_samples)
