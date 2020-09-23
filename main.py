import os
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, 'relu', input_shape = (784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

model = create_model()

model.summary()

checkpoint_path = "training_1"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save(checkpoint_path)

