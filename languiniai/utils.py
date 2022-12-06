import numpy as np
import librosa
import os

def load_wavs(wavs: np.ndarray) -> np.ndarray:
    return np.ndarray( list(map( lambda file: librosa.load(f'{dir}/{file}'), os.listdir(dir) )) )

