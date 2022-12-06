import numpy as np
import librosa
import math

def rms(wav: np.ndarray) -> np.ndarray:
    '''calculate audio rms'''
    return math.sqrt( np.mean( wav**2 ) )
