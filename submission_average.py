import pandas as pd
import numpy as np


pred1 = pd.read_csv('predictions/submission2_xgb_1165_89.csv')
pred2 = pd.read_csv('predictions/submission3_xgb_1164_78.csv')

gm = (pred1['loss']*pred2['loss'])**(1/2.)
#am = (pred1['loss']+pred2['loss']+pred3['probability']+pred4['probability'])/4.

d = {'id':pred1['id'],
     'loss':gm
     }

submit = pd.DataFrame(data=d,columns=['id','loss'])
submit.to_csv('predictions/submission4_ave_gm_sub2_sub3.csv',index=False)
