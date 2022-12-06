import requests
import time
import json
import os
from datetime import datetime

FORVO_API_KEY = os.environ['FORVO_API_KEY']
BUCKET_NAME   = os.environ['BUCKET_NAME']

TODAY = datetime.utcnow().strftime('%Y%m%d')

ROOT_DIR = f'/media/zdata/23-data_science/lewagon_ubt/code/matthias-403/gcp_bucket/forvo_api/{TODAY}'

from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def listdir_nohidden(path: str = './') -> list:
    '''list directory excluding hidden file'''
    return [ f for f in os.listdir(path) if not f.startswith('.') ]

import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
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
    return (np.asarray(AudioSegment.from_file(full_name, format='mp3', frame_rate = sample_rate).get_array_of_samples(), dtype = np.float64), sample_rate)

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
    
from scipy.io.wavfile import write
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

# redundant, will eventually uses jonathan's code
# ================================================================
import numpy as np
import math
def rms(wav: np.ndarray) -> np.ndarray:
    '''calculate audio rms'''
    return math.sqrt( np.mean( wav**2 ) )

# https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
def awgn_noise(original_wav: np.ndarray, scale: float = 1) -> np.ndarray:
    STD_n = rms(original_wav) * scale
    noise = np.random.normal(0, STD_n, original_wav.shape[0])
    return original_wav + noise

import random
def real_world_noise(original_wav: np.ndarray, real_noise: np.array) -> np.ndarray:
    start_sample = random.randrange(0, len(real_noise))
    noise = np.array( [ real_noise[sample % len(real_noise)] for sample in range(start_sample, start_sample + len(original_wav) + 1) ] )
    return (original_wav + noise) / 2

import matplotlib.pyplot as plt
import librosa
import librosa.display
def save_spectrogram(audio: np.ndarray, sample_rate: int = 44100, target_full_name: str = '') -> None:
    '''convert 1 single audio np.array to spectrogram, file saved locally'''
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure()
    librosa.display.specshow(S_dB)
    plt.savefig(f'{target_full_name}.png')
    plt.close()
    print(f'spectrogram saved: {target_full_name}.png')

# ================================================================

import json
def top_words(file: str) -> list:
    """Read top words file on GCP bucket, return a list of words"""
    b = bucket.blob(file)
    with b.open("r") as f:
        return list(json.load(f).values())

# request_url = f'https://apifree.forvo.com/key/{FORVO_API_KEY}/format/json/action/popular-pronounced-words/language/en'
def api_daily_call(words: list = []) -> None:
    """call forvo api and save response"""
    forvo_api_url = 'https://apifree.forvo.com'
    format        = 'format/json'
    language      = 'language/en'  # https://forvo.com/languages-codes/

    # gets all the pronunciations from a word.
    # https://api.forvo.com/documentation/word-pronunciations/
    action  = 'action/word-pronunciations'
    country = 'country/gbr'   # https://en.wikipedia.org/wiki/ISO_3166-1
    order   = 'order/rate-desc' # date-desc, date-asc, rate-desc, rate-asc
    limit   = 'limit/50'

    os.system(f'mkdir -p {ROOT_DIR}/json')
    
    def requests_operation(word: str) -> None:
        url = f'{forvo_api_url}/key/{FORVO_API_KEY}/{format}/{action}/word/{word}/{language}/{country}/{order}/{limit}'
        response = requests.get(url, timeout=10).json()
        with open(f'{ROOT_DIR}/json/{word}.json', 'w') as f:
            json.dump(response, f)
            print(url)
        
    map(requests_operation, words)
      
# re-write when there is free time, very ugly function
def api_daily_generate_spectrogram_png() -> None:
    '''
    load mp3 files of the day and process into spectrogram
    '''
    load_dir_list = [     
                          f'{ROOT_DIR}/mp3/speaker1'
                        , f'{ROOT_DIR}/mp3/speaker2'
                        , f'{ROOT_DIR}/mp3/single_speaker' 
                    ]

    target_dir_list = [   
                          [ f'{ROOT_DIR}/png/original/speaker1'       , f'{ROOT_DIR}/png/awgn_noise/speaker1'       ]#, f'{ROOT_DIR}/jpg/real_world_noise/speaker1' ]
                        , [ f'{ROOT_DIR}/png/original/speaker2'       , f'{ROOT_DIR}/png/awgn_noise/speaker2'       ]#, f'{ROOT_DIR}/jpg/real_world_noise/speaker2' ]
                        , [ f'{ROOT_DIR}/png/original/single_speaker' , f'{ROOT_DIR}/png/awgn_noise/single_speaker' ]#, f'{ROOT_DIR}/jpg/real_world_noise/single_speaker' ]
                    ]
    
    for load_dir, target_dir in zip(load_dir_list, target_dir_list):
        name_list = listdir_nohidden(load_dir)
        audios = [ load_mp3(f'{load_dir}/{file}') for file in name_list  ]
        for i, target in enumerate(target_dir):
            os.system(f'mkdir -p {target}')
            target_full_name = [ f"{target}/{file.replace('.mp3','')}" for file in name_list ]
            match i:
                case 0: [ save_spectrogram(                 audio[0] , audio[1], full_name) for audio, full_name in zip(audios, target_full_name) ]
                case 1: [ save_spectrogram(      awgn_noise(audio[0]), audio[1], full_name) for audio, full_name in zip(audios, target_full_name) ]
                case 2: [ save_spectrogram(real_world_noise(audio[0]), audio[1], full_name) for audio, full_name in zip(audios, target_full_name) ]

def test_func():
    pass
        
if __name__ == "__main__":
    
    
    #top_words_file = 'forvo_api/top_forvo.json'
    #start = 0
    #count = 450
    #all_words = top_words(top_words_file)
    #words = [ all_words[word % len(all_words)] for word in range(start, start + count + 1) ]
    
    #print( api_daily_call(words) )
    api_daily_generate_spectrogram_png()

