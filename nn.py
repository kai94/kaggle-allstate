import numpy as np
import pandas as pd
from keras.models import Model,Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras.layers.core import Dense,Flatten,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,Callback
from keras import backend as K
from sklearn.cross_validation import KFold,train_test_split
import os

def nn():
    input = Input(shape=(1190,))
    out = Dense(500)(input)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Dropout(0.5)(out)
    out = Dense(50)(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Dropout(0.3)(out)
    out = Dense(1)(out)
    model = Model(input=input,output=out)
    model.compile(loss='mae',
                  optimizer=Adam(lr=1e-4))
    return model



if __name__ == '__main__':
    np.random.seed(2016)
    random_state = 2016

    X = np.load('data/X3.npy')
    y = np.load('data/y.npy')
    X_pred = np.load('data/X3_pred.npy')
    """
    X_cont = X[:,:14]
    X_cont_mean = np.mean(X_cont,axis=0)
    X_cont_std = np.mean(X_cont,axis=0)
    X_cont_norm = (X_cont - X_cont_mean) / X_cont_std
    X[:,:14] = X_cont_norm
    np.save('data/X3.npy',X)
    X_pred_cont = X_pred[:,:14]
    X_pred_norm = (X_pred_cont - np.mean(X_pred_cont,axis=0)) / np.std(X_pred_cont,axis=0)
    X_pred[:,:14] = X_pred_norm
    """
    np.save('data/X3_pred.npy',X_pred)
    kf = KFold(len(y), 5, False, random_state)
    predictions = []
    for train_idx, test_idx in kf:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model = nn()
        weight_path = 'weights.h5'
        callbacks = [
            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ]
        print model.summary()
        model.fit(X_train, y_train, batch_size = 64,nb_epoch=120,
                  validation_data = (X_test, y_test),callbacks=callbacks)
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        pred = model.predict(X_pred)
        predictions.append(pred)
    predictions = np.array(predictions)
    np.save('predictions/X_pred_nn.npy',predictions)
