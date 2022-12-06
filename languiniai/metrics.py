import numpy as np
import librosa
import math

def noise_rms(y):
    '''
    y: signal from librosa.load(filepath) tuple
    '''
    return math.sqrt(np.mean(y**2))
