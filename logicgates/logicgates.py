from keras.layers import Input, Dense
from keras.models import Model

import numpy as np

def logic_gate(labels,
               input=np.array([[0., 0.],
                               [0., 1.],
                               [1., 0.],
                               [1., 1.]])):
    inputs = Input(shape=(2,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(2048, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(input, labels)

    return model

def xor_gate():
    return logic_gate(np.array([[1., 0.], [0., 1.], [0., 1.], [1., 0.]]))

def and_gate():
    return logic_gate(np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]]))

def or_gate():
    return logic_gate(np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.]]))

def not_gate():
    inputs = Input(shape=(1,))
    x = Dense(2, activation='softmax')(inputs)
    predictions = Dense(1, activation='relu')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(np.array([[0.], [1.]]), np.array([[1.], [0.]]))

    return model

if __name__ == '__main__':
    xor = xor_gate()
    print xor.predict(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]))

    notfcn = not_gate()
    print notfcn.predict(np.array([[0.], [1.]]))