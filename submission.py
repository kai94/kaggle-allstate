import pandas as pd
import numpy as np

idx = np.load('data/idx.npy')
pred = np.load('data/pred2.npy')


print pred.shape
pred = np.mean(pred,axis=0)
d = {'id':pd.Series(idx),
     'loss':pd.Series(pred)
     }

submit = pd.DataFrame(data=d,columns=['id','loss'])
submit.to_csv('predictions/submission2_xgb_1165_89.csv',index=False)
