# taskB/model.py
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class L2NormalizationLayer(tf.keras.layers.Layer):
    """Performs L2 normalization."""
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

@tf.keras.utils.register_keras_serializable()
class EuclideanDistanceLayer(tf.keras.layers.Layer):
    """Calculates the L2 distance between two vectors."""
    def call(self, inputs):
        vec1, vec2 = inputs
        sum_square = tf.reduce_sum(tf.square(vec1 - vec2), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def get_embedding_model(image_shape, embedding_dim):
    """Creates the base CNN model for generating image embeddings."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=image_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    inputs = tf.keras.layers.Input(shape=image_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(embedding_dim, activation=None)(x)
    x = L2NormalizationLayer()(x)
    return tf.keras.Model(inputs, x, name="EmbeddingModel")

def get_siamese_model(embedding_model, image_shape):
    """Creates the full Siamese Network."""
    input_anchor = tf.keras.layers.Input(name="anchor", shape=image_shape)
    input_comparison = tf.keras.layers.Input(name="comparison", shape=image_shape)
    
    embedding_anchor = embedding_model(input_anchor)
    embedding_comparison = embedding_model(input_comparison)
    
    distance = EuclideanDistanceLayer()([embedding_anchor, embedding_comparison])
    prediction = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
    return tf.keras.Model([input_anchor, input_comparison], prediction, name="SiameseNetwork")

CUSTOM_OBJECTS = {
    'L2NormalizationLayer': L2NormalizationLayer,
    'EuclideanDistanceLayer': EuclideanDistanceLayer
}