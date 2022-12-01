import numpy as np
import librosa
import math

from os import listdir

def noise_rms(y):
    '''
    y: signal from librosa.load(filepath) tuple
    '''
    return math.sqrt(np.mean(y**2))

def snoise(y, scale=1):
    '''
    y: signal from librosa.load(filepath) tuple,
    scale helps us scale the noise down and adjust it to our use
    '''
    
    rms = noise_rms(y)
    return np.random.normal(0, rms*scale, y.shape[0])

def with_noise(y_real, scale=1):
    '''
    y_real: signal we add noise to, the real data
    scale: the scale that goes into the snoise() function
    '''
    return y_real + snoise(y_real, scale)

def real_world_noises(source_path: str, noise_path: str) -> np.ndarray:
    y1, sample_rate1 = librosa.load(source_path, mono=True)
    y2, sample_rate2 = librosa.load(noise_path, mono=True)
    return (y1 + y2[0:44100])/2
