#  Languini AI
## Your new AI study budy for languages!

Languini is trained on thousands of images of spectrograms of real-world speakers. It uses siamese twins with triplets loss function to train a model to recognise similar sounding sounds, by comparing their spectrograms.




***
### How it works
The basic idea is very simple: 

1. Voice is recorded as .WAV and then converted to an analogue signal
2. The analogue signal is converted to a Mel Spectrogram using a fast fourier transform (FFT) (input, for the model)
3. Spectrogram of the recording is compared with the spectrogram of the native speaker (anchor, for the model)
4. The model embedes both images (as TensorFlow vectors)and compares them through cosine similarity
5. The resulting score is weighted against previous attemps to show progress

![alt text](https://github.com/Matthias-403/Languini_AI/blob/master/GH%20diagram(3).png)

### Reusing Languini AI
#### Training 
Create a directory of files with audiorecordings of sounds in the target language. You can generate a model using the languiniai/train_model.py 
#### Comparing
You can use our train model and languiniai/compare.py to compare two different melspectrograms.
#### Data Pre-processing
Within the notebooks branch there is a directory with jupyter notebooks that walk through it all.
***
