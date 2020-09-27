# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:49:49 2020

@author: Mert Ketenci
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:14:29 2020

@author: Mert Ketenci
"""

import pickle
from items import Items
from feature2id import Feature2Id
import numpy as np
from tqdm import tqdm

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def one_hot_encode(items,feature2id):
    
    item_size = feature2id.v_size()
    w_size = feature2id.w_size()
    a_size = feature2id.a_size()
    g_size = feature2id.g_size()
    c_size = feature2id.c_size()
    p_size = feature2id.p_size()
    d_size = feature2id.d_size()
    u_size = feature2id.u_size()
    s_size = feature2id.s_size()
    
    w_matrix = np.zeros((item_size,w_size), dtype=np.int16)
    a_matrix = np.zeros((item_size,a_size), dtype=np.int16)
    g_matrix = np.zeros((item_size,g_size), dtype=np.int16)
    c_matrix = np.zeros((item_size,c_size), dtype=np.int16)
    p_matrix = np.zeros((item_size,p_size), dtype=np.int16)
    d_matrix = np.zeros((item_size,d_size), dtype=np.int16)
    s_matrix = np.zeros((item_size,s_size), dtype=np.int16)
    
    for i in tqdm(range(item_size)):
        for x in items.WM[i]:
            if x!=0:
                w_matrix[i][x] +=1
        for x in items.AM[i]:
            if x!=0:
                a_matrix[i][x] +=1
        for x in items.GM[i]:
            if x!=0:
                g_matrix[i][x] +=1
        if items.CM[i]!=0:
            c_matrix[i][x] +=1
        for x in items.PM[i]:
            if x!=0:
                p_matrix[i][x] +=1
        for x in items.DM[i]:
            if x!=0:
                d_matrix[i][x] +=1
        for x in items.SM[i]:
            if x!=0:
                s_matrix[i][x] +=1
                
    features = np.concatenate([w_matrix,a_matrix,g_matrix,c_matrix,p_matrix,d_matrix,s_matrix],axis = 1)
    features = sparse.csr_matrix(features)    

    return features

path = './'

with open(path+'items.pk', 'rb') as fd:
    items = pickle.load(fd)
with open(path+'feature2id.pk', 'rb') as fd:
    feature2id = pickle.load(fd)

item_features = one_hot_encode(items,feature2id)


ratings =  sparse.coo_matrix(items.rating_matrix_tr_tr[:,:-4000] + items.rating_matrix_va_tr[:,:-4000] + items.rating_matrix_te_tr[:,:-4000])

train = items.rating_matrix_tr_tr[:-20000,:-4000]
# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.

# Instantiate and train the model
model = LightFM(loss='warp', no_components=60)
model.fit(ratings,
          item_features=item_features[:-4000],
          epochs=10)