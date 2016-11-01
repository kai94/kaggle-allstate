from sklearn.linear_model import LinearRegression,Lasso, HuberRegressor, SGDRegressor,PassiveAggressiveRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
import random
from sklearn.cross_validation import train_test_split, KFold,cross_val_score
from sklearn.metrics import mean_absolute_error
from keras.models import Model,Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras.layers.core import Dense,Flatten,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU, PReLU
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,Callback
from keras import backend as K
from sklearn.cross_validation import KFold,train_test_split
import os


def nn():
    input = Input(shape=(11,))
    out = Dense(10)(input)
    out = ELU()(out)
    out = Dense(1)(out)
    model = Model(input=input,output=out)
    model.compile(loss='mae',
                  optimizer=Adam())
    return model
if __name__=='__main__':
    np.random.seed(2016)
    random_state = 2016

    pred1 = np.load('blend/blend_pred_nn_3_1141.npy').squeeze()
    pred2 = np.load('blend/blend_pred_xgb_1_1133.npy')
    pred3 = np.load('blend/blend_pred_xgb_2_fair_1138.npy')
    pred4 = np.load('blend/blend_pred_lsvr_1.npy')
    pred5 = np.load('blend/blend_pred_par_1.npy')



    train1 = np.load('blend/blend_train_nn_3_1141.npy').squeeze()
    train2 = np.load('blend/blend_train_xgb_1_1133.npy')
    train3 = np.load('blend/blend_train_xgb_2_fair_1138.npy')
    train4 = np.load('blend/blend_train_lsvr_1.npy')
    train5 = np.load('blend/blend_train_par_1.npy')


    y = np.load('data/y.npy')
    """
    X_pred = np.vstack((pred1, pred2, pred3, pred4, 
                        pred5, pred6, pred7, pred8, 
                        pred10, pred11)).transpose()
    X = np.vstack((train1, train2, train3, train4, 
                   train5, train6, train7, train8, 
                   train10, train11)).transpose()
    """
    X_pred = np.vstack((pred1, pred2, pred3, pred4,
                        pred5)).transpose()
    X = np.vstack((train1, train2, train3, train4,
                   train5)).transpose()

    print X.shape



    
    kf = KFold(len(y), 5, False, random_state)
    predictions = []
    for train_idx, test_idx in kf:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        """
        model = nn()
        weight_path = 'weights.h5'
        callbacks = [
            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ]
        print model.summary()
        model.fit(X_train, y_train, batch_size = 64,nb_epoch=60,
                  validation_data = (X_test, y_test),callbacks=callbacks)
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        pred = model.predict(X_pred)
        predictions.append(pred)
        """
        clf = HuberRegressor()
        clf.fit(X_train, y_train)
        #clf.fit(X, y)
        print mean_absolute_error(y_test, clf.predict(X_test))
        pred = clf.predict(X_pred)
        predictions.append(pred)

    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = random_state)
    
        
    #clf = HuberRegressor(max_iter=500,
    #                    epsilon = 1.0)
    #clf = HuberRegressor()
    #clf.fit(X_train, y_train)
    #clf.fit(X, y)
    #print mean_absolute_error(y_test, clf.predict(X_test))
    #predictions = clf.predict(X_pred)
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    np.save('predictions/pred_blend_huber.npy',predictions)

