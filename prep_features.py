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


    #
    # Remove unnecessary categories
    #
    X['cat75'][X['cat75'] == 'C'] = 'B'
    X['cat77'][X['cat77'] == 'A'] = 'B'
    X['cat78'][X['cat78'] == 'C'] = 'A'
    X['cat78'][X['cat78'] == 'D'] = 'A'
    X['cat79'][X['cat79'] == 'C'] = 'A'
    X['cat80'][X['cat80'] == 'A'] = 'C'
    X['cat81'][X['cat81'] == 'A'] = 'C'
    X['cat85'][X['cat85'] == 'A'] = 'C'
    X['cat85'][X['cat85'] == 'D'] = 'C'
    X['cat86'][X['cat86'] == 'A'] = 'C'
    X['cat88'][X['cat88'] == 'B'] = 'E'
    X['cat89'][X['cat89'] == 'D'] = 'C'
    X['cat89'][X['cat89'] == 'E'] = 'C'
    X['cat89'][X['cat89'] == 'H'] = 'C'
    X['cat89'][X['cat89'] == 'G'] = 'C'
    X['cat89'][X['cat89'] == 'F'] = 'C'
    X['cat89'][X['cat89'] == 'I'] = 'C'
    X['cat90'][X['cat90'] == 'D'] = 'C'
    X['cat90'][X['cat90'] == 'E'] = 'C'
    X['cat90'][X['cat90'] == 'F'] = 'C'
    X['cat90'][X['cat90'] == 'G'] = 'C'
    X['cat91'][X['cat91'] == 'E'] = 'D'
    X['cat91'][X['cat91'] == 'F'] = 'D'
    X['cat91'][X['cat91'] == 'H'] = 'D'
    X['cat92'][X['cat92'] == 'C'] = 'B'
    X['cat92'][X['cat92'] == 'I'] = 'B'
    X['cat92'][X['cat92'] == 'D'] = 'B'
    X['cat92'][X['cat92'] == 'E'] = 'B'
    X['cat92'][X['cat92'] == 'G'] = 'B'
    X['cat92'][X['cat92'] == 'F'] = 'B'
    X['cat93'][X['cat93'] == 'E'] = 'B'
    X['cat93'][X['cat93'] == 'A'] = 'B'
    X['cat94'][X['cat94'] == 'F'] = 'A'
    X['cat94'][X['cat94'] == 'E'] = 'A'
    X['cat94'][X['cat94'] == 'G'] = 'A'
    X['cat95'][X['cat95'] == 'B'] = 'A'
    X['cat96'][X['cat96'] == 'A'] = 'F'
    X['cat96'][X['cat96'] == 'C'] = 'F'
    X['cat96'][X['cat96'] == 'I'] = 'F'
    X['cat96'][X['cat96'] == 'H'] = 'F'
    X['cat97'][X['cat97'] == 'F'] = 'D'
    X['cat97'][X['cat97'] == 'B'] = 'D'
    X['cat99'][X['cat99'] == 'C'] = 'E'
    X['cat99'][X['cat99'] == 'J'] = 'E'
    X['cat99'][X['cat99'] == 'H'] = 'E'
    X['cat99'][X['cat99'] == 'M'] = 'E'
    X['cat99'][X['cat99'] == 'I'] = 'E'
    X['cat99'][X['cat99'] == 'G'] = 'E'
    X['cat99'][X['cat99'] == 'O'] = 'E'
    X['cat99'][X['cat99'] == 'U'] = 'E'
    X['cat100'][X['cat100'] == 'M'] = 'O'
    X['cat100'][X['cat100'] == 'C'] = 'O'
    X['cat100'][X['cat100'] == 'D'] = 'O'
    X['cat100'][X['cat100'] == 'E'] = 'O'
    X['cat101'][X['cat101'] == 'S'] = 'O'
    X['cat101'][X['cat101'] == 'R'] = 'O'
    X['cat101'][X['cat101'] == 'E'] = 'O'
    X['cat101'][X['cat101'] == 'B'] = 'O'
    X['cat101'][X['cat101'] == 'H'] = 'O'
    X['cat101'][X['cat101'] == 'K'] = 'O'
    X['cat101'][X['cat101'] == 'U'] = 'O'
    X['cat101'][X['cat101'] == 'N'] = 'O'
    X['cat102'][X['cat102'] == 'D'] = 'E'
    X['cat102'][X['cat102'] == 'G'] = 'E'
    X['cat102'][X['cat102'] == 'F'] = 'E'
    X['cat102'][X['cat102'] == 'H'] = 'E'
    X['cat102'][X['cat102'] == 'J'] = 'E'
    X['cat103'][X['cat103'] == 'G'] = 'F'
    X['cat103'][X['cat103'] == 'H'] = 'F'
    X['cat103'][X['cat103'] == 'I'] = 'F'
    X['cat103'][X['cat103'] == 'J'] = 'F'
    X['cat103'][X['cat103'] == 'L'] = 'F'
    X['cat103'][X['cat103'] == 'K'] = 'F'
    X['cat103'][X['cat103'] == 'N'] = 'F'
    X['cat103'][X['cat103'] == 'M'] = 'F'
    
    cat = X['cat104'].value_counts()
    for index in cat.index[10:]:
        X['cat104'][X['cat104'] == index] = 'J'

    cat = X['cat105'].value_counts()
    for index in cat.index[7:]:
        X['cat105'][X['cat105'] == index] = 'J'

    cat = X['cat106'].value_counts()
    for index in cat.index[9:]:
        X['cat106'][X['cat106'] == index] = 'D'

    cat = X['cat107'].value_counts()
    for index in cat.index[9:]:
        X['cat107'][X['cat107'] == index] = 'D'

    cat = X['cat108'].value_counts()
    for index in cat.index[9:]:
        X['cat108'][X['cat108'] == index] = 'H'

    cat = X['cat104'].value_counts()
    for index in cat.index[10:]:
        X['cat104'][X['cat104'] == index] = 'J'

    cat = X['cat109'].value_counts()
    for index in cat.index[5:]:
        X['cat109'][X['cat109'] == index] = 'G'

    cat = X['cat110'].value_counts()
    for index in cat.index[18:]:
        X['cat110'][X['cat110'] == index] = 'BS'

    cat = X['cat111'].value_counts()
    for index in cat.index[6:]:
        X['cat111'][X['cat111'] == index] = 'K'

    cat = X['cat112'].value_counts()
    for index in cat.index[21:]:
        X['cat112'][X['cat112'] == index] = 'O'

     cat = X['cat113'].value_counts()
    for index in cat.index[20:]:
        X['cat113'][X['cat113'] == index] = 'J'

    cat = X['cat114'].value_counts()
    for index in cat.index[7:]:
        X['cat114'][X['cat114'] == index] = 'I'

    cat = X['cat115'].value_counts()
    for index in cat.index[10:]:
        X['cat115'][X['cat115'] == index] = 'H'

    cat = X['cat116'].value_counts()
    for index in cat.index[16:]:
        X['cat116'][X['cat116'] == index] = 'LN'











    cats_onehot = ['cat1',
                   'cat2',
                   'cat4',
                   'cat5',
                   'cat6',
                   'cat9',
                   'cat10',
                   'cat11',
                   'cat12',
                   'cat13',
                   'cat23',
                   'cat25'
                   'cat27',
                   'cat36',
                   'cat37',
                   'cat38',
                   'cat44',
                   'cat50',
                   'cat53',
                   'cat71',
                   'cat72',
                   'cat73',
                   'cat75',
                   'cat76',
                   'cat77',
                   'cat78',
                   'cat79',
                   'cat80',
                   'cat81',
                   'cat82',
                   'cat83',
                   'cat84',
                   'cat85',
                   "cat86",
                    "cat87",
                    "cat88",
                    "cat89",
                    "cat90",
                    "cat91",
                    "cat92",
                    "cat93",
                    "cat94",
                    "cat95",
                    "cat96",
                    "cat97",
                    "cat98",
                    "cat99",
                    "cat100",
                    "cat101",
                    "cat102",
                    "cat103",
                    "cat104",
                    "cat105",
                    "cat106",
                    "cat107",
                    "cat108",
                    "cat109",
                    "cat110",
                    "cat111",
                    "cat112",
                    "cat113",
                    "cat114",
                    "cat115",
                    "cat116"
                   ]
                   
                   
                   
    cats_popularity = ['cat3',
                       'cat7',
                       'cat8',
                       'cat14',
                       'cat15',
                       'cat16',
                       'cat17',
                       'cat18',
                       'cat19',
                       'cat20',
                       'cat21',
                       'cat22',
                       'cat24',
                       'cat26',
                       'cat28',
                       'cat29',
                       'cat30',
                       'cat31',
                       'cat32',
                       'cat33',
                       'cat34',
                       'cat35',
                       'cat39',
                       'cat40',
                       'cat41',
                       'cat42',
                       'cat43',
                       'cat45',
                       'cat46',
                       'cat47',
                       'cat48',
                       'cat49',
                       'cat51',
                       'cat52',
                       'cat54',
                       'cat55',
                       'cat56',
                       'cat57',
                       'cat58',
                       'cat59',
                       'cat60',
                       'cat61',
                       'cat62',
                       'cat63',
                       'cat64',
                       'cat65',
                       'cat66',
                       'cat67',
                       'cat68',
                       'cat69',
                       'cat70',
                       'cat74',
                       
                       
                       
                       

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
    np.save('data/X4.npy',X)


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
