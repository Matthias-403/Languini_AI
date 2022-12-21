#  Languini AI
## Your new AI study budy for languages!

Languini is trained on thousands of images of spectrograms of real-world speakers. It uses siamese twins with triplets loss function to train a model to recognise similar sounding sounds, by comparing their spectrograms.


### Reusing Languini AI
We recommed loading our pretrain model to play around with, the loaded model acts as the embedding for the images (look at compare.py), which works with cosine similarity between two images.

***
## Basic Idea
Fast Fourier Transforms (FFTs) work very well on small samples of data, so the idea was to compare the FFTs of users' speach with native speakers.
To portray the voice in a best way possible using the time axis as well, mel-spectrogram using Librosa library were applied to .wav files.
The images where then compared using siamese model.

