from tensorflow.keras.applications import resnet
from tensorflow.keras import metrics
import gcp_bucket as gb

import tensorflow as tf
from tensorflow import expand_dims
from tensorflow import float32 as tf_float32

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


# downloads the images for the comparing
#local paths
#im1 = preprocess_image('/home/mp/code/Matthias-403/project-local/my_voice_recordings/sg/cat_char.jpg')
#im2 = preprocess_image('/home/mp/code/Matthias-403/project-local/my_voice_recordings/sg/cat_pos.jpg')
#im3 = preprocess_image('/home/mp/code/Matthias-403/project-local/my_voice_recordings/sg/cat_chris.jpg')


#local paths
#embedding1 = tf.keras.models.load_model('forvo_1.h5')
#cloud paths
#embedding = gb.load_files(['trained_model/test.h5'])

#similarity between 2 images (im2,im3)  relative to the anchor (im1)
#sim = similarity_3(im1,im2,im3, embedding)

#print results
#print(sim)

#prints the model from cloud and its type
#print(type(embedding1[0]), type(embedding1))
