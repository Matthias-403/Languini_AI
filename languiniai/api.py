import io
import json
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa.display
from librosa.feature import melspectrogram

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pytz

from pydub import AudioSegment
from pydub.utils import mediainfo

from google.cloud import storage

import add_receive

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Data(BaseModel):
    word : str
    user : str

embedding = add_receive.load_model('model.h5')

@app.post("/get_example")
def get_example(data: Data = Depends()):
    request_params = json.loads(data.json())
    
    storage_client = storage.Client()
    bucket = storage_client.bucket('languini-ai-bucket')
    blob_list = [ blob.name for blob in bucket.list_blobs() ]
    
    matches = [ match for match in blob_list if f"/{request_params['word']}.mp3" in match ]
    
    blob = bucket.blob(matches[0])
    
    def iterfile(blob):
        with blob.open("rb") as file_like:
            yield from file_like
            
    return StreamingResponse(iterfile(blob), media_type="audio/mp3")

@app.post("/get_result")
def get_result(data: Data = Depends(), file: UploadFile = File(...)):
    
    request_params = json.loads(data.json())
    audio = file.file.read()
    
    spectrogram_png_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-tmp.png"
    y, sr = librosa.load(io.BytesIO(audio))[0], 44100
    S = melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure()
    librosa.display.specshow(S_dB)
    plt.savefig(spectrogram_png_name)
    plt.close()
    
    user = request_params['user']
    word = request_params['word']
    
    mff_rtn = add_receive.mff(spectrogram_png_name, user, word, embedding)
    
    mff_rtn['scores'] = [ float(score) for score in mff_rtn['scores'] ]
    mff_rtn['dates']  = [ int(date)    for date  in mff_rtn['dates'] ]
    mff_rtn['good']   = [ 1. ]
    mff_rtn['bad']    = [ 0.9937421093999812 ]
    
    os.remove(spectrogram_png_name)
    
    return mff_rtn

