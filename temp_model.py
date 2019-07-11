#coding=utf-8

try:
    import findspark
except:
    pass

try:
    from pyspark import SparkContext, SparkConf
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.optimizers import SGD
except:
    pass

try:
    from elephas.utils.rdd_utils import to_simple_rdd
except:
    pass

try:
    import keras
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from elephas.spark_model import SparkModel
except:
    pass

try:
    import keras
except:
    pass

try:
    from hyperopt import STATUS_OK
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.optimizers import RMSprop
except:
    pass

try:
    from elephas.hyperparam import HyperParamModel
except:
    pass

try:
    from pyspark import SparkContext, SparkConf
except:
    pass

from __future__ import print_function

try:
    from __future__ import unicode_literals
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.datasets import mnist
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
nb_classes = 10
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


def keras_fmin_fnct(space):


    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(space['Dropout']))
    model.add(Dense(space['Dense']))
    model.add(Activation('relu'))
    model.add(Dropout(space['Dropout_1']))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    model.fit(x_train, y_train,
              batch_size=space['batch_size'],
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model.to_yaml()}

def get_space():
    return {
        'Dropout': hp.uniform('Dropout', 0, 1),
        'Dense': hp.choice('Dense', [256, 512, 1024]),
        'Dropout_1': hp.uniform('Dropout_1', 0, 1),
        'batch_size': hp.choice('batch_size', [64, 128]),
    }
