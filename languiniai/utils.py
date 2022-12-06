import os
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo
from scipy.io.wavfile import write

def load_mp3(full_name: str) -> tuple:
    '''
    input (full_name): file full name(<path>/<file_name>.<extension>)
    output (pcm: np.ndarray, int: sample_rate):
        - float64 pcm data
        - sample rate
    load locally stored .mp3 file to pcm samples as np.ndarray
    '''
    print(f"loading '{full_name}'")
    sample_rate = int(mediainfo(full_name).get('sample_rate', 44100))
    return np.asarray(AudioSegment.from_file(full_name, format='mp3', frame_rate = sample_rate).get_array_of_samples(), dtype = np.float64), sample_rate

def load_wav(full_name: list) -> tuple:
    '''
    input [full_name]: file full name(<path>/<file_name>.<extension>)
    output(pcm: np.ndarray, int: sample_rate):
        - float64 pcm data
        - sample rate
    load locally stored .wav file to pcm samples as np.ndarray
    '''
    print(f"loading '{full_name}'")
    sample_rate = int(mediainfo(full_name).get('sample_rate', 44100))
    return np.asarray(AudioSegment.from_file(full_name, format='wav', frame_rate = mediainfo(full_name)['sample_rate']).get_array_of_samples(), dtype = np.float64), sample_rate

def pcm_to_wav(pcm: np.ndarray, sample_rate: int, target_full_name: str) -> str:
    '''
    input (pcm): raw pcm data as np.ndarray
    input (sample_rate): sample rate in integer
    input (target_full_name): target location with file name and extension
    output (target_full_name): return target_full_name
    export .wav file from raw pcm data
    '''
    write(target_full_name, sample_rate, pcm.astype(np.int16))
    return target_full_name

def generate_noise_sg(file_name:str, source_dir:str, target_dir:str, scale:float, save=False):
    '''
    generates a spectrogram with added noise
    
    file_name: name of the file 
    source_dir: source directory for the .wav files
    target_dir: target directory spectrograms are saved
    scale: scaling factor that scales the RMS in the noise (part of snoise())
    save: when unchanged it displays the image, when true it saves it to the target directory
    '''
    
    wav_name = source_dir + '/' + file_name
    y, sr = librosa.load(wav_name)[0], 44100
    ny = with_noise(y,scale)
    S = librosa.feature.melspectrogram(y=ny, sr=sr)
    
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure()
    librosa.display.specshow(S_dB)

    if save==True:
        pic_name = target_dir + '/' + file_name.replace('.wav','.jpg')
        plt.savefig(pic_name)
        plt.close()
    else:
        plt.show()
    
def save_all_noise_sg(source_dir,target_dir, scale):    
    '''
    takes all the files in the source_dir and generates their spectrograms inside
    target_dir, scale scales RMS in the noise (part of snoise())
    '''
    
    splits_list = listdir(source_dir)
    for filename in splits_list:
        generate_noise_sg(filename, source_dir, target_dir, scale, save=True)
