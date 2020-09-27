# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:31:38 2020

@author: Mert Ketenci
"""
from collections import defaultdict

class Feature2Id:
    def __init__(self):
        self.id2item = {0:'<PAD>'}
        self.item2id = {'<PAD>':0}
        self.v_idx = 0
        
        self.id2wrter = {0:'<PAD>'}
        self.wrter2id = {'<PAD>':0}
        self.w_idx = 1
                
        self.id2actor = {0:'<PAD>'}
        self.actor2id = {'<PAD>':0}
        self.a_idx = 1
        
        self.id2genre = {0:'<PAD>'}
        self.genre2id = {'<PAD>':0}
        self.g_idx = 1
        
        self.id2colle = {0:'<PAD>'}
        self.colle2id = {'<PAD>':0}
        self.c_idx = 1
        
        self.id2produ = {0:'<PAD>'}
        self.produ2id = {'<PAD>':0}
        self.p_idx = 1
        
        self.id2direc = {0:'<PAD>'}
        self.direc2id = {'<PAD>':0}
        self.d_idx = 1
        
        self.id2summ = {0:'<PAD>'}
        self.summ2id = {'<PAD>':0}
        self.s_idx = 1
                        
    def push_name(self,token,movieId2id):
        name = token['title']
        id = token['movieId']
        self.item2id[name] = movieId2id[id]
        self.id2item[movieId2id[id]] = name
        self.v_idx += 1
    
    def push_metadata(self,item_metadata):
        for i in range(item_metadata.shape[0]):
            writer = item_metadata.iloc[i]['writer']
            actor = item_metadata.iloc[i]['actor']
            genre = item_metadata.iloc[i]['genre']
            collection = item_metadata.iloc[i]['collection']
            production = item_metadata.iloc[i]['production']
            director = item_metadata.iloc[i]['director']
            summary = item_metadata.iloc[i]['summary']
            
            for w in writer: 
                if w not in self.wrter2id.keys():
                    self.wrter2id[w] = self.w_idx
                    self.id2wrter[self.w_idx] = w
                    self.w_idx += 1 
                    
            for a in actor:    
                if a not in self.actor2id.keys():
                    self.actor2id[a] = self.a_idx
                    self.id2actor[self.a_idx] = a
                    self.a_idx += 1
                    
            for g in genre:  
                if g not in self.genre2id.keys():
                    self.genre2id[g] = self.g_idx
                    self.id2genre[self.g_idx] = g
                    self.g_idx += 1
        
            if collection not in self.colle2id.keys():
                self.colle2id[collection] = self.c_idx
                self.id2colle[self.c_idx] = collection
                self.c_idx += 1   
                
            for p in production:
                if p not in self.produ2id.keys():
                    self.produ2id[p] = self.p_idx
                    self.id2produ[self.p_idx] = p
                    self.p_idx += 1
                    
            for d in director:
                if d not in self.direc2id.keys():
                    self.direc2id[d] = self.d_idx
                    self.id2direc[self.d_idx] = d
                    self.d_idx += 1
            
            for s in summary:
                if s not in self.summ2id.keys():
                    self.summ2id[s] = self.s_idx
                    self.id2summ[self.s_idx] = s
                    self.s_idx += 1
    
    def push_train_item(self, train_items) :
        self.train_items = train_items
        
    def push_test_item(self, test_items):
        self.test_items = test_items
        
    def push_train_user(self,token,userId2id):
        token = token['userId'].unique()
        self.train_users = [userId2id[x] for x in token]
    
    def push_test_user(self,token,userId2id):
        token = token['userId'].unique()
        self.test_users = [userId2id[x] for x in token]
    
    def push_valid_user(self,token,userId2id):
        token = token['userId'].unique()
        self.valid_users = [userId2id[x] for x in token]
    
    def train_item_size(self):
        return len(self.train_items)
    
    def test_item_size(self):
        return len(self.test_items)
    
    def train_user_size(self):
        return len(self.train_users)
    
    def test_user_size(self):
        return len(self.test_users)
    
    def valid_user_size(self):
        return len(self.valid_users)
    
    def get_train_items(self):
        return self.train_items
    
    def get_test_items(self):
        return self.test_items
    
    def get_train_users(self):
        return self.train_users
    
    def get_test_users(self):
        return self.test_users
    
    def get_valid_users(self):
        return self.valid_users
    
    def id2v(self,id):
        return self.id2item[id]
    
    def v2id(self,id):
        return self.item2id[id]
    
    def id2w(self,id):
        return self.id2wrter[id]
    
    def w2id(self,token):
        return self.wrter2id[token]
            
    def id2a(self,id):
        return self.id2actor[id]
    
    def a2id(self,token):
        return self.actor2id[token]

    def id2g(self,id):
        return self.id2genre[id]
    
    def g2id(self,token):
        return self.genre2id[token]
    
    def id2c(self,id):
        return self.id2colle[id]
    
    def c2id(self,token):
        return self.colle2id[token]
    
    def id2p(self,id):
        return self.id2produ[id]
    
    def p2id(self,token):
        return self.produ2id[token]
    
    def id2d(self,id):
        return self.id2direc[id]
    
    def d2id(self,token):
        return self.direc2id[token]
    
    def id2s(self,id):
        return self.id2summ[id]
    
    def s2id(self,token):
        return self.summ2id[token]
    
    def w_size(self):
        return self.w_idx
        
    def a_size(self):
        return self.a_idx
    
    def g_size(self):
        return self.g_idx
    
    def c_size(self):
        return self.c_idx
    
    def p_size(self):
        return self.p_idx
    
    def d_size(self):
        return self.d_idx
    
    def s_size(self):
        return self.s_idx
    
    def v_size(self):
        return self.v_idx
    
    def u_size(self):
        return len(self.train_users) + len(self.test_users) + len(self.valid_users)
