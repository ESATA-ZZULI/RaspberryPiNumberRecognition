import tensorflow as tf
import numpy as np
from PIL import Image

def loadMyData():
    res = []
    for i in range(1, 9):
        img = Image.open(f"./newData/{i}.bmp")
        arr = np.asarray(img, dtype=np.float32).reshape(784)
        print(arr.shape)
        res.append(arr)

    return np.array(res), np.array(range(1, 9))

(x_train, y_train) = loadMyData()
x_train = x_train / 255.0
x_test = x_train
y_test = y_train


try:
    model = tf.keras.models.load_model('./workspace/xnewm.h5')
except:
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Dense(, input_shape=(1, 784)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)
model.evaluate(x_test, y_test)

model.save('./workspace/xnewm.h5')
