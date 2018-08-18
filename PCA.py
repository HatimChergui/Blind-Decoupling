# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 10:17:47 2018

@author: Administrator
"""

import pandas as pd 
#import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


path = 'Dataset_Blind_Decoupling_All.xls'
Dataset = pd.read_excel(path)


x = Dataset.iloc[:,0:10].values
y = Dataset.iloc[:,10].values

x = StandardScaler().fit_transform(x)

pca = PCA(svd_solver='full', n_components=3)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents , columns = ['PC1', 'PC2', 'PC3'])

finalDf = pd.concat([principalDf, Dataset['Class']], axis = 1)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('PC1', fontsize = 11)
ax.set_ylabel('PC2', fontsize = 11)
ax.set_zlabel('PC3', fontsize = 11)
ax.set_title('3 Component PCA', fontsize = 11)
Classes = [1, 2, 3, 4]
colors = ['r', 'g', 'b', 'cyan']
for target, color in zip(Classes,colors):
    indicesToKeep = Dataset['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1'] , finalDf.loc[indicesToKeep, 'PC2'], finalDf.loc[indicesToKeep, 'PC3'], c = color, cmap="Set2_r", s=60)
ax.legend(Classes)
ax.grid()
plt.savefig('PCA.eps', format='eps', dpi=1000)