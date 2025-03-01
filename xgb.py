import numpy as np
import pandas as pd
import xgboost as xgb
import random
from sklearn.cross_validation import train_test_split, KFold,cross_val_score
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.svm import LinearSVR,SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from fastFM.als import FMRegression
from scipy.sparse import csc_matrix


if __name__=='__main__':
    np.random.seed(2016)
    random_state = 2016
    random.seed(2016)


    X = np.load('data/X3.npy')
    y = np.load('data/y.npy')
    n = 188318
    X_pred = X[n:]
    n1,n2 = X_pred.shape
    string = type('a')
    NAN = []
    count = 0
    for i in range(n1):
        for j in range(n2):
            if type(X_pred[i,j]) == string:
                NAN.append(X_pred[i,j])
                count += 1
    print count
    NAN = set(NAN)
    for nan in NAN:
        X_pred[X_pred == nan] = 0.0

    X_pred = np.asfarray(X_pred)
    X = X[:n]
    np.save('data/X4_train.npy',X)
    np.save('data/X4_pred.npy',X_pred)
    
    X = np.load('data/X3_train.npy')
    X_pred = np.load('data/X3_pred.npy')
    y = np.load('data/y.npy')

    predictions = []
    kf = KFold(len(X),5,random_state=random_state)
    val_loss = []

    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        """
        clf = xgb.XGBRegressor(max_depth=10,
                               learning_rate=0.015, 
                                   n_estimators=1000, 
                                   silent=True, 
                                   objective='reg:linear', 
                                   nthread=-1, 
                                   gamma=0,
                                   min_child_weight=0.15,
                                   max_delta_step=0, 
                                   subsample=0.5,
                                   colsample_bytree=0.6,
                                   colsample_bylevel=1, 
                                   reg_alpha=0, 
                                   reg_lambda=1, 
                                   scale_pos_weight=1, 
                                   seed=1440, 
                                   missing=None)
        
        res = clf.fit(X_train, y_train, eval_metric='mae', verbose = True, 
                      eval_set = [(X_test, y_test)],early_stopping_rounds=200)
        fig, ax = plt.subplots(1, 1, figsize=(7, 30))
        xgb.plot_importance(clf,ax=ax)

        plt.savefig('feature_importance_xgb.png')
        E = res.best_score
        print E
        val_loss.append(E)
        pred = clf.predict(X_pred)
        predictions.append(pred)
        """
        fm = FMRegression(n_iter=100, init_stdev=0.1, rank=10, l2_reg_w=0.1, l2_reg_V=0.5)
        fm.fit(csc_matrix(X_train), y_train)
        E = mean_absolute_error(y_test,fm.predict(csc_matrix(X_test)))
        print E
        val_loss.append(E)
        pred = fm.predict(csc_matrix(X_pred))
        predictions.append(pred)
    params={'max_depth': [4,6,8,10],
            'subsample': [0.5,0.75,1],
            'colsample_bytree': [0.4, 0.6,0.8,1.0],
            'learning_rate':[0.015],
            'objective':['reg:linear'],
            'seed':[1440],
            'min_child_weight':[0.15,0.30,0.5,1],
            'n_estimators':[1000]
    }
    """
    clf = xgb.XGBRegressor()        
    gs = RandomizedSearchCV(clf,params,cv=3,scoring='mean_absolute_error',verbose=2,n_iter=13)
    gs.fit(X,y)
    print gs.best_score_
    print gs.best_estimator_
    """


    print val_loss
    predictions = np.array(predictions)
    np.save('data/pred3',predictions)
