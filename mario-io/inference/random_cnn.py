import tensorflow as tf

def build_random_policy_model(input_shape=(90, 160, 3)) -> tf.keras.Model:
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 5, strides=2, activation="relu")(inp)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # Output lx, ly in [-1, 1]
    out = tf.keras.layers.Dense(2, activation="tanh")(x)
    return tf.keras.Model(inp, out)
