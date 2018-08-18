# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:49:30 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

path = 'Dataset_Blind_Decoupling_All.xls'
Dataset = pd.read_excel(path)

N_training_max = 3000

Training_Set = Dataset.iloc[:N_training_max,:]
np.random.seed(0)
indices = np.arange(N_training_max)
np.random.shuffle(indices)
Training_Set = Training_Set.iloc[indices]
decoupling_success_rate  = []
acu                      = []
for n in range(50,N_training_max+1,50):
   # np.random.seed(0)
    X_train = Training_Set.iloc[:n,0:10]
    y_train = Training_Set.iloc[:n,10]
#    indices = np.arange(y.shape[0])
#    np.random.shuffle(indices)
#    X_train, y_train = X.iloc[indices], y.iloc[indices]

    acu_score_avg = []
    dsr_avg = []
    X_test  = Dataset.iloc[N_training_max+1:Dataset.shape[0],0:10]
    y_test  = Dataset.iloc[N_training_max+1:Dataset.shape[0],10]

# Standardization of the training set
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
# ML algorithms
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', C=1, cache_size=1000, gamma=0.4, random_state=0, decision_function_shape='ovo')
    svm.fit(X_train_std, y_train)

    T = 200 # 5 MRs, 1MR every 10 ms
    N = (Dataset.shape[0] - N_training_max)//T

    acu_score = np.zeros((1,N))

    for i in range(N):
        X_test_std = sc.transform(X_test.iloc[i*T :(i+1)*T, :])
        y_pred = svm.predict(X_test_std)
        acu_score[:,i] = accuracy_score(y_test[i*T :(i+1)*T], y_pred, normalize=True, sample_weight=None)

    acu_score_avg.append((acu_score).mean())
    dsr_avg.append((acu_score>=0.9).mean())
    sc = np.array(acu_score_avg, np.float)
    dsr = np.array(dsr_avg, np.float)
    decoupling_success_rate.append(np.mean(dsr))
    acu.append(np.mean(sc))
    

    
# plotting the decoupling success rate
import matplotlib.pyplot as plt
from scipy import interpolate


plt.figure()
plt.grid(linestyle='-', linewidth='0.5', color='gray')
plt.xticks(np.arange(50, 3001, step=300))
plt.yticks(np.arange(0.1, 1.1, step=0.05))
plt.ticklabel_format(style='sci', scilimits=(0,1), useMathText='True')
n_split = np.arange(50,N_training_max+1,50)
tck1 = interpolate.splrep(n_split, decoupling_success_rate, s=0)
tck2 = interpolate.splrep(n_split, acu, s=0)
xnew = np.arange(50,N_training_max+1,50)
xnew2 = np.arange(50,N_training_max+1,10)
ynew1 = interpolate.splev(xnew, tck1, der=0)
ynew2 = interpolate.splev(xnew, tck2, der=0)
threshold_decision_success = 0.73*np.ones((60))

plt.plot(xnew, ynew1,'b', label="SVM with RBF")
plt.plot(xnew, threshold_decision_success,'--m', label="Fixed Thresholds")
plt.plot(xnew, ynew2, color='coral', label='Accuracy Score')

plt.xlabel('Training Samples')
plt.ylabel('Decoupling Success Rate')
plt.legend()
plt.show
plt.savefig('Decoupling_Success_rate_v5.eps', format='eps', dpi=1000)
















