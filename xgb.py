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
if __name__=='__main__':
    np.random.seed(2016)
    random_state = 2016
    random.seed(2016)
    """
    X_train = pd.read_csv('data/train.csv')
    X_pred = pd.read_csv('data/test.csv')
    y = X_train['loss'].values
    
    X_train.drop(['id','loss'], axis=1, inplace = True)
    idx = X_pred['id'].values
    np.save('data/y.npy',y)
    np.save('data/idx.npy',idx)
    X_pred.drop(['id'], axis = 1, inplace = True)
    print len(X_train),len(X_pred)
    X = pd.concat((X_train,X_pred)).reset_index(drop=True)

    n_train = len(X_train)
    features = X.columns
    cats = [feat for feat in features if 'cat' in feat]
    cats_onehot = cats[:76]
    cats_popularity = cats[76:]

    for feat in cats_onehot:
        dat = X[feat]
        name = dat.value_counts().index
        dat.columns = [feat + n for n in name]
        X.drop([feat], axis = 1, inplace = True)
        dat = pd.get_dummies(dat)
        X = pd.concat([X,dat], axis = 1)

    for feat in cats_popularity:
        dat = X[:n_train][feat]
        pop = dat.value_counts() / n_train
        dat = X[feat]
        for i in pop.index:
            dat[dat == i] = pop[i]
        X[feat] = dat
        

    print X
    X = X.as_matrix()
    np.save('data/X.npy',X)
        
    """
    X = np.load('data/X.npy')
    y = np.load('data/y.npy')
    n = 188318
    X_pred = X[n:]
    n1,n2 = X_pred.shape
    string = type('a')
    NAN = []
    for i in range(n1):
        for j in range(n2):
            if type(X_pred[i,j]) == string:
                NAN.append(X_pred[i,j])
               
    NAN = set(NAN)
    for nan in NAN:
        X_pred[X_pred == nan] = 0.0

    X_pred = np.asfarray(X_pred)
    X = X[:n]



    
    predictions = []
    kf = KFold(len(X),5,random_state=random_state)
    val_loss = []
    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        """
        clf = xgb.XGBRegressor(max_depth=6,
                            learning_rate=0.02, 
                                   n_estimators=1000, 
                                   silent=True, 
                                   objective='reg:linear', 
                                   nthread=-1, 
                                   gamma=0,
                                   min_child_weight=0.1,
                                   max_delta_step=0, 
                                   subsample=0.8, 
                                   colsample_bytree=1.,
                                   colsample_bylevel=1, 
                                   reg_alpha=0, 
                                   reg_lambda=1, 
                                   scale_pos_weight=1, 
                                   seed=1440, 
                                   missing=None)
        
        res = clf.fit(X_train, y_train, eval_metric='mae', verbose = True, 
                      eval_set = [(X_test, y_test)],early_stopping_rounds=200)
        xgb.plot_importance(clf)
        plt.savefig('feature_importance_xgb.png')
        print res.best_score
        """
        """
        clf = KNeighborsRegressor(n_neighbors=50,n_jobs=-1)
        clf.fit(X_train,y_train)
        E = mean_absolute_error(y_test,clf.predict(X_test))
        print E
        val_loss.append(E)
        pred = clf.predict(X_pred)
        predictions.append(pred)
        """
    params={'max_depth': [4,6,8,10],
            'subsample': [0.5,0.75,1],
            'colsample_bytree': [0.4, 0.6,0.8,1.0],
            'learning_rate':[0.01,0.006,0.002],
            'objective':['binary:logistic'],
            'seed':[1440],
            'min_child_weight':[0.15,0.30,0.5,1],
            'n_estimators':[1000]
    }

    clf = xgb.XGBRegressor()        
    gs = RandomizedSearchCV(clf,params,cv=4,scoring='mae',verbose=2,n_iter=30)
    gs.fit(dat,target)
    print gs.best_score_
    print gs.best_estimator_



    print val_loss
    predictions = np.array(predictions)
    np.save('data/pred',predictions)
