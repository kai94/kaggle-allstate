import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import itertools


shift = 200.

COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(',')



def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i] )-ord('A')+1)*26**(ln-i-1)
    return r


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def xg_eval_mae_exp(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(y, yhat)


def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    # compute skew and do Box-Cox transformation (Tilli)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


if __name__ == "__main__":
    directory = 'data/'
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')
    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    cats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    ntrain = len(train)

    #train_test, ntrain = mungeskewed(train, test, numeric_feats)

    for feat in cats:
        if train[feat].nunique() != test[feat].nunique():
            set_train = set(train[feat].unique())
            set_test = set(test[feat].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)
            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x

            train_test[feat] = train_test[feat].apply(lambda x: filter_cat(x), 1)
        
        count = train_test[feat].value_counts() / len(train_test)

        """
        remove = count[count < 0.005].index.tolist()
        def filter_cat(x):
            if x in remove:
                return "XXX"
            return x
        train_test[feat] = train_test[feat].apply(lambda x: filter_cat(x), 1)
        """



    print len(train_test)
    cats = [x for x in train.columns[1:-1] if 'cat' in x]
    for comb in itertools.combinations(COMB_FEATURE, 2):
        feat = comb[0] + "_" + comb[1]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
        train_test[feat] = train_test[feat].apply(encode)
        #print(feat)
    cats = [x for x in train.columns[1:-1] if 'cat' in x]        
    
    """
    for comb in itertools.combinations(COMB_STRONG, 2):
        feat = comb[0] + "_" + comb[1]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
        train_test[feat] = train_test[feat].apply(encode)
        #print(feat)
    cats = [x for x in train.columns[1:-1] if 'cat' in x]  
    """
    for col in cats:
        train_test[col] = train_test[col].apply(encode)

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index
    
    for feat in skewed_feats:
        train_test[feat] += 1
        train_test[feat], lam = boxcox(train_test[feat])

    ss = StandardScaler()
    #train_test[numeric_feats] = ss.fit_transform(train_test[numeric_feats].values)
    train = train_test.iloc[:ntrain, :].copy()    
    y = train['loss'].values
    train.drop('loss',inplace=True,axis=1)
    #train = train.as_matrix()
    test = train_test.iloc[ntrain:, :].copy()
    test.drop('loss', inplace=True, axis=1)
    #test = test.as_matrix()
    train.drop('id',inplace=True,axis=1)
    test.drop('id',inplace=True,axis=1)

    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.01,
        #'objective': 'reg:linear',
        'max_depth': 12,
        'min_child_weight': 100,
        'booster': 'gbtree',
        #'base_score':2000.
    }

    params2 = {
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': 0,
        'base_score':7.
    }

    y_log = np.log(y + shift)
    y_4 = (y+1)**0.25

    best_nrounds = 20000  # 640 score from above commented out code (Faron)
    allpredictions = pd.DataFrame()
    kfolds = 10
    predictions = []


    kf = KFold(len(train), n_folds=kfolds)

    blend_train = np.zeros(len(train))
    dtest = xgb.DMatrix(test)
    val_loss = []
    for i, (train_index, test_index) in enumerate(kf):
        print('Fold {0}'.format(i + 1))
        X_train = train.iloc[train_index]
        X_val = train.iloc[test_index]
        y_train = y_log[train_index]
        y_val = y_log[test_index]
        dtrain = xgb.DMatrix(X_train,
                             label=y_train)
        dvalid = xgb.DMatrix(X_val,
                             label=y_val)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbdt = xgb.train(params2, dtrain, best_nrounds, watchlist,
                         obj=fair_obj,
                         feval=xg_eval_mae_exp, 
                         maximize=False,
                         verbose_eval=50,
                         early_stopping_rounds=50)

        pred = np.exp(gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)) - shift
        #pred = gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)
        #pred = (gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit))**4 - 1.
        predictions.append(pred)

        pred = np.exp(gbdt.predict(dvalid, ntree_limit=gbdt.best_ntree_limit)) - shift
        #pred = gbdt.predict(dvalid, ntree_limit=gbdt.best_ntree_limit)
        #pred = (gbdt.predict(dvalid, ntree_limit=gbdt.best_ntree_limit))**4 - 1.
        blend_train[test_index] = pred
        E = mean_absolute_error(np.exp(y_val) - shift, pred)
        print E
        val_loss.append(E)


    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    print sum(val_loss) / kfolds
    np.save('blend_pred_xgb_20_lex_fair3_c07_log',predictions)
    np.save('blend_train_xgb_20_lex_fair3_c07_log',blend_train)
