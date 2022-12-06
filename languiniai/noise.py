import numpy as np
import metrics

# https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
def awgn_noise(wav: np.ndarray, scale: float =1) -> np.ndarray:
    '''t'''
    STD_n = rms(wav) * scale
    noise = np.random.normal(0, STD_n, wav.shape[0])
    return wav + noise

def real_world_noises(original_wav: np.ndarray, real_noise: np.array) -> np.ndarray:
    return (original_wav + len(original_wav[0])) / 2
