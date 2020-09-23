import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


model_path = "training_1"
model_dir = os.path.dirname(model_path)

checkpoint_path = "training_2"
checkpoint_dir = os.path.dirname(checkpoint_path)


model = tf.keras.models.load_model(model_path)


# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)

# 새로운 콜백으로 모델 훈련하기
model.fit(train_images, 
          train_labels,  
          epochs=100,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 콜백을 훈련에 전달합니다