import numpy as np
import metrics

def snoise(y, scale=1):
    '''
    y: signal from librosa.load(filepath) tuple,
    scale helps us scale the noise down and adjust it to our use
    '''
    
    rms = noise_rms(y)
    return np.random.normal(0,rms*scale,y.shape[0])

def with_noise(y_real, scale=1):
    '''
    y_real: signal we add noise to, the real data
    scale: the scale that goes into the snoise() function
    '''
    return y_real+snoise(y_real,scale)

def real_world_noises(original_wav: np.ndarray, real_noise: np.array) -> np.ndarray:
    return (original_wav + len(original_wav[0])) / 2

