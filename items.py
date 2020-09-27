# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:04:58 2020

@author: Mert Ketenci
"""
import numpy as np
from scipy import sparse

class Items:
    def __init__(self, item_metadata, feature2id):
        
        item_size = feature2id.v_size()
        
        self.WS = max([len(x) for x in item_metadata['writer']])
        self.AS = max([len(x) for x in item_metadata['actor']])
        self.GS = max([len(x) for x in item_metadata['genre']])
        self.PS = max([len(x) for x in item_metadata['production']])
        self.DS = max([len(x) for x in item_metadata['director']])
        self.SS = max([len(x) for x in item_metadata['summary']])
        
        self.WM = np.zeros((item_size, self.WS),dtype = np.int32)
        self.AM = np.zeros((item_size, self.AS),dtype = np.int32)
        self.GM = np.zeros((item_size, self.GS),dtype = np.int32)
        self.CM = np.zeros((item_size, ),dtype = np.int32)
        self.PM = np.zeros((item_size, self.PS),dtype = np.int32)
        self.DM = np.zeros((item_size, self.DS),dtype = np.int32)
        self.SM = np.zeros((item_size, self.SS),dtype = np.int32)
        
    def push_metadata(self,item_metadata,feature2id):
        for i in range(item_metadata.shape[0]):
            id = item_metadata.iloc[i]['ids']
            writer = item_metadata.iloc[i]['writer']
            actor = item_metadata.iloc[i]['actor']
            genre = item_metadata.iloc[i]['genre']
            collection = item_metadata.iloc[i]['collection']
            production = item_metadata.iloc[i]['production']
            director = item_metadata.iloc[i]['director']
            summary = item_metadata.iloc[i]['summary']
            
            if id != '<PAD>':
                self.WM[id][:len(writer)] = [feature2id.wrter2id[x] for x in writer]
                self.AM[id][:len(actor)] = [feature2id.actor2id[x] for x in actor]   
                self.GM[id][:len(genre)] = [feature2id.genre2id[x] for x in genre]
                self.CM[id] = feature2id.colle2id[collection]
                self.PM[id][:len(production)] =[feature2id.produ2id[x] for x in production]
                self.DM[id][:len(director)] =[feature2id.direc2id[x] for x in director]
                self.SM[id][:len(summary)] =[feature2id.summ2id[x] for x in summary]
                
    def push_rating_matrix_train(self,ratings_tr,ratings_te,feature2id):
        self.rating_matrix_tr_tr = ratings_tr
        self.rating_matrix_tr_te = ratings_te
        self.rating_matrix_tr = ratings_tr + ratings_te
        
    def push_rating_matrix_valid(self,ratings_tr,ratings_te,feature2id):
        self.rating_matrix_va_tr = ratings_tr
        self.rating_matrix_va_te = ratings_te
        
    def push_rating_matrix_test(self,ratings_tr,ratings_te,feature2id):
        self.rating_matrix_te_tr = ratings_tr
        self.rating_matrix_te_te = ratings_te
    
    def get_rating_matrix_tr(self,ids,axis = 'u'):
        if axis == 'u':
            return self.rating_matrix_tr[ids].todense()
        elif axis =='v':
            return self.rating_matrix_tr.T[ids].todense()

    def get_rating_matrix_tr_tr(self,ids,axis = 'u'):
        if axis == 'u':
            return self.rating_matrix_tr_tr[ids].todense()
        elif axis =='v':
            return self.rating_matrix_tr_tr.T[ids].todense()

    def get_rating_matrix_tr_te(self,ids,axis = 'u'):
        if axis == 'u':
            return self.rating_matrix_tr_te[ids].todense()
        elif axis =='v':
            return self.rating_matrix_tr_te.T[ids].todense()
        
    def get_rating_matrix_va_tr(self,ids,axis='u'):
        if axis == 'u':
            return self.rating_matrix_va_tr[ids].todense()
        elif axis =='v':
            return self.rating_matrix_va_tr.T[ids].todense()
    
    def get_rating_matrix_va_te(self,ids,axis='u'):
        if axis == 'u':
            return self.rating_matrix_va_te[ids].todense()
        elif axis =='v':
            return self.rating_matrix_va_te.T[ids].todense()
        
    def get_rating_matrix_te_tr(self,ids,axis='u'):
        if axis == 'u':
            return self.rating_matrix_te_tr[ids].todense()
        elif axis =='v':
            return self.rating_matrix_te_tr.T[ids].todense()
        
    def get_rating_matrix_te_te(self,ids,axis='u'):
        if axis == 'u':
            return self.rating_matrix_te_te[ids].todense()
        elif axis =='v':
            return self.rating_matrix_te_te.T[ids].todense()
        
    def get_w(self,ids):
        return self.WM[ids]
    
    def get_s(self,ids):
        return self.SM[ids]

    def get_a(self,ids):
        return self.AM[ids]
    
    def get_g(self,ids):
        return self.GM[ids]
    
    def get_c(self,ids):
        return self.CM[ids]
    
    def get_p(self,ids):
        return self.PM[ids]
    
    def get_d(self,ids):
        return self.DM[ids]
    
    def get_u(self,ids):
        return self.UM[ids]