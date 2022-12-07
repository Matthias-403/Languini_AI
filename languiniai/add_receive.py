from tensorflow.keras.applications import resnet
from tensorflow.keras import metrics
from google.cloud import storage
import os
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow import float32 as tf_float32
from datetime import datetime
import json
import pandas as pd
#data preproc, model loading and prediction
def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape. designed to read local file,
    needs work to see how it will behave when on streamlit
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (400,400))

    return image

def load_model(h5_model_filepath):
    '''
    loads the local model
    '''
    return tf.keras.models.load_model(h5_model_filepath)

def embed_image(sample, embedding):
    '''
    takes in the preprocessed image (sample) and embedds it based
    on a preloaded embedding .
    '''
    embedded_image= embedding(resnet.preprocess_input(\
        expand_dims(sample, 0)))

    return embedded_image

def similarity_2(sample_1, sample_2, embedding):
    '''
    produces similarity between two photos, results is a float
    '''
    cosine_similarity = metrics.CosineSimilarity()
    similarity = cosine_similarity(embed_image(sample_1, embedding), embed_image(sample_2, embedding))

    return similarity.numpy()

def similarity_3(sample_1, sample_2, sample_3, embedding):
    '''
    similarity between 3 groups, against the 1st entry,
    results is a tuple: (1 compared to 2, 1 compared to 3)
    '''
    cosine_similarity = metrics.CosineSimilarity()
    similarity12 = similarity_2(sample_1, sample_2, embedding)
    similarity13 = similarity_2(sample_1, sample_3, embedding)

    return similarity12,similarity13

#CSV Functions
def get_df_cloud(file_path:str)->object:
    '''
    gets csv file from the bucket and returns it as a pandas dataframe
    '''
    return pd.read_csv(file_path)

def user_word(df:object, user_name:str, word:str)-> object:
    '''
    returns a pandas dataframe sorted by 'attempt', with columns:
    user, word, attempt, score, timestamp
    func specifies the word, so the df is for one word only.
    '''
    try:
        return df[(df.user==user_name) & (df.word ==word)].sort_values(by=['attempt'])
    except:
        return 0

def base_score(df:object, user_name:str, word:str)->float:
    '''
    returns the first score ever for the given word,
    otherwise returns zero
    '''
    if word in df[(df.user==user_name)].word.to_numpy():
        return df[(df.user == user_name) & (df.word == word)].sort_values(by=['attempt']).score.to_numpy()[0]
    else:
        return 0

def get_scores(df:object, user_name:str, word:str, get_dates:bool=False)->list:
    '''
    gets a list of scores for a given user and and word,
    if get_dates=True, returns a tuple of lists, format:
    (scores, dates)
    '''
    try:
        if not get_dates:
            return (user_word(df, user_name, word).score.to_numpy(),[])
        if get_dates:
            return (user_word(df, user_name, word).score.to_numpy(),
                    user_word(df, user_name,word).timestamp.to_numpy())
    except:
        return ([],[])

def upload_csv(df:str,target_name:str='users_scores2.csv',bucket_name:str='languini-ai-bucket',
               bucket_dir_path:str='user_scores')->None:
    '''
    saves the df: dataframe as a csv locally and saves it as target_name='name.csv',
    target name has to include the extension, bucket_name and bucket_dir_path are predefined
    but can ge changed to be repurposed. If theres an error a mesasge is returned.
    this function returns nothing
    '''
    try:
        df.to_csv(f'./{target_name}', index=False)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{bucket_dir_path}/{target_name}")
        blob.upload_from_filename(f'./{target_name}')
        os.remove(target_name)
    except:
        return 'ERROR with upload_csv(), nothing uploaded'

def new_row(df:object, user:str, word:str,score:str=0, upload:bool=False)->object:
    '''
    adds a new row on the input dataframe df, and uploads it to the filepath
    put to upload_csv(), or the default one (check upload_csv() for details)

    the function creates a new row for new word or new row incremented by +1
    for the attempt
    '''
    df_l = df[df.user==user]
    current_attempt=1
    now = datetime.now()
    datetime_str = now.strftime("%y%m%d%H%M")
    if word in df_l[(df_l.user==user)].word.to_numpy():
        current_attempt = df_l[(df_l.user==user) & (df_l.word==word)].\
            attempt.sort_values().iloc[0] + 1

    new_row ={'user':user, 'word':word, 'attempt':current_attempt,
              'score':score, 'timestamp': datetime_str}
    df2 = df.append(new_row, ignore_index=True)
    if not upload:
        return df2
    if upload:
        upload_csv(df2)
        return df2


def mff(sg_filepath: str,user: str, word: str, embedding:object)->dict:
    '''
    takes in the local path of the spectrogram of users input as the path to a
    librosa melspectrogram jpg (sg_filepath), the user name to be used (user),
    the word being tested (word, used to import the path to the cloud .jpg of
    the spectrogram for that word, need a locally saved .json with paths)
    and the loaded model for the embedding of the images (embedding, must be a
    <class 'keras.engine.functional.Functional'> object), the function updates
    the csv with user performance data and returns a dict of information for
    the streamlit site
    '''

    #local path to the file, for optimal speed. either saved locally on the VM
    #or burned on the container image
    f = open('/home/mp/code/Matthias-403/project-local/notebooks/words_paths.json')
    dict_load = json.load(f)
    #im1 is the image from user we test
    im1 = preprocess_image(sg_filepath)
    word_path = dict_load.get(word,0)
    if word_path==0:
        return 0
    #im2 is th anchor
    im2 = preprocess_image(f"gs://languini-ai-bucket/{word_path}")
    #cosine similarity fo the two words
    score=similarity_2(im1,im2,embedding)
    #updating the cscv
    csv_filepath='gs://languini-ai-bucket/user_scores/users_scores2.csv'
    df = get_df_cloud(csv_filepath)
    df = new_row(df, user, word, score, upload=True)
    #summary of scores, scores are a list of all scores for the word
    #dates are a list of all attempts per given word
    scores, dates = get_scores(df, user, word, get_dates=True)
    return {'user': user, 'word': word, 'scores':scores, 'dates':dates}


##---------------------------------------------------------------------
# EXAMPLES
##---------------------------------------------------------------------

# cloud model
# embedding1 = load_model('gs://languini-ai-bucket/trained_model/2212042016_MP.h5')

# local model
# embedding2 = load_model("/home/mp/code/Matthias-403/project-local/models/forvo_2.h5")

# calling the mff()
# a =mff('/home/mp/code/Matthias-403/project-local/my_voice_recordings/sg/cat_pos.jpg',
        # 'User_1','cat',embedding2)
