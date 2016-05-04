from keras.layers import Input, Dense
from keras.models import Model

import numpy as np

data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
labels = np.array([[0.], [1.], [1.], [0.]])

def xor_gate():
    inputs = Input(shape=(2,))
    x = Dense(2, activation='relu')(inputs)
    predictions = Dense(1, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, labels)

    return model

if __name__ == '__main__':
    xor = xor_gate()
    print xor.predict(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]))