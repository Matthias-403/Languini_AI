import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
import gcp_bucket as gb



def siamese_model(anchor_images: list, positive_images: list, save_model_name: str,\
    n_epochs: int, n_batches: int, cloud:bool = False):

    def preprocess_image(filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        return image

    def preprocess_triplets(anchor, positive, negative):
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """

        return (
            preprocess_image(anchor),
            preprocess_image(positive),
            preprocess_image(negative),
        )



    target_shape = (400,400)

    #cloud dependant change later

    image_count = len(anchor_images)

    # this will be changed to return a tf_dataset from Google Cloud
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)


    # To generate the list of negative images, let's randomize the list of
    # available images and concatenate them together.
    rng = np.random.RandomState(seed=42)
    rng.shuffle(anchor_images)
    rng.shuffle(positive_images)

    negative_images = anchor_images + positive_images
    np.random.RandomState(seed=32).shuffle(negative_images)

    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
    negative_dataset = negative_dataset.shuffle(buffer_size=4096)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(8)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(8)


    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable


    class DistanceLayer(layers.Layer):
        """
        This layer is responsible for computing the distance between the anchor
        embedding and the positive embedding, and the anchor embedding and the
        negative embedding.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, anchor, positive, negative):
            ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
            an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
            return (ap_distance, an_distance)

    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    class SiameseModel(Model):
        """The Siamese Network model with a custom training and testing loops.

        Computes the triplet loss using the three embeddings produced by the
        Siamese Network.

        The triplet loss is defined as:
        L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
        """

        def __init__(self, siamese_network, margin=0.5):
            super(SiameseModel, self).__init__()
            self.siamese_network = siamese_network
            self.margin = margin
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        def call(self, inputs):
            return self.siamese_network(inputs)

        def train_step(self, data):
            # GradientTape is a context manager that records every operation that
            # you do inside. We are using it here to compute the loss so we can get
            # the gradients and apply them using the optimizer specified in
            # `compile()`.
            with tf.GradientTape() as tape:
                loss = self._compute_loss(data)

            # Storing the gradients of the loss function with respect to the
            # weights/parameters.
            gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

            # Applying the gradients on the model using the specified optimizer
            self.optimizer.apply_gradients(
                zip(gradients, self.siamese_network.trainable_weights)
            )

            # Let's update and return the training loss metric.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):
            loss = self._compute_loss(data)

            # Let's update and return the loss metric.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def _compute_loss(self, data):
            # The output of the network is a tuple containing the distances
            # between the anchor and the positive example, and the anchor and
            # the negative example.
            ap_distance, an_distance = self.siamese_network(data)

            # Computing the Triplet Loss by subtracting both distances and
            # making sure we don't get a negative value.
            loss = ap_distance - an_distance
            loss = tf.maximum(loss + self.margin, 0.0)
            return loss

        @property
        def metrics(self):
            # We need to list our metrics here so the `reset_states()` can be
            # called automatically.
            return [self.loss_tracker]


    def modeling(batchSize, n_epochs):
        siamese_model = SiameseModel(siamese_network)
        siamese_model.compile(optimizer=optimizers.Adam(0.0001))
        siamese_model.fit(train_dataset,batch_size=batchSize, epochs=n_epochs, validation_data=val_dataset)


    ### move that into a function
    def save_model(filename):
        base_cnn.save(filename)

    modeling(n_batches, n_epochs)

    # testing it locally needs to upload to cloud too
    if not cloud:
        save_model(save_model_name)

    if cloud:
        # need to save model to cloud
        pass


# make work with gcloud storage
def get_images_list(images_path: str, n_start: int, n_end:int,\
    cloud: bool=False) -> list:

    if not cloud:
        images = [str(images_path+'/'+f) for f in os.listdir(anchor_images_path)][n_start:n_end]
        return images

    if cloud:
        #get cloud images from path
        return list(map(lambda x: f"gs://{os.getenv('BUCKET_NAME')}/{x}",\
            gb.list_files(images_path)[n_start:n_end]))




anchor_images_path   = 'forvo_api/20221201/jpg/original/1st_speaker'
positive_images_path = 'forvo_api/20221201/jpg/original/2nd_speaker'

anchor_images = get_images_list(anchor_images_path, 0, 200, cloud=True)
positive_images = get_images_list(positive_images_path, 0, 200, cloud=True)



siamese_model(anchor_images, positive_images,"forvo_2.h5", 4, 64)

print('Complete!')
