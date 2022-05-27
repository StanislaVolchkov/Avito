from typing import Tuple, Union
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pymorphy2

import catboost
#import optuna
#from sklearn.model_selection import train_test_split
from navec import Navec
nltk.download("stopwords")
nltk.download('punkt')
stop = set(stopwords.words("russian"))

class task1():
    
    def __init__(self, df):
        self.data = df
        self.embeddings_pretrained = Navec.load('lib/navec_hudlit_v1_12B_500K_300d_100q.tar')
        
    def make_right_data(self):
        
        self.morph = pymorphy2.MorphAnalyzer()
        cat_change = dict.fromkeys(list(self.data['subcategory'].unique()))
        for i in cat_change.keys():
            cat_change[i] = round(self.data[self.data['subcategory'] == i]['price'].mean(), 1)
            self.data.loc[(self.data['subcategory'] == i) & (self.data['price'].isna() == True), 'price'] = cat_change[i]
    
        self.data['description'] = self.data['description'].apply(lambda x: x.lower())
        self.data['description'] = self.data['description'].apply(lambda x: re.sub(r'[,\.]',' ', x)) 
        self.data['description'] = self.data['description'].apply(lambda x: re.sub(r'[_]','', x)) 
        self.data['description'] = self.data['description'].apply(lambda x: re.sub(r'[^\w\s]','', x)) 
        self.data['description'] = self.data['description'].apply(lambda x: self.preproc(x)) 
    
    def preproc(self, text):
        
        return ' '.join([list(self.morph.parse(token))[0].normal_form for token in word_tokenize(text) if token not in stop])
    
    def vectorize_sum(self, text, embeddings):
        
        embedding_dim = 300
        features = np.zeros([embedding_dim], dtype='float32')

        for word in text.split():
            if word in embeddings:
                features += embeddings[f'{word}']
    
        return features
    
    @property
    def train(self):

        train, test = train_test_split(self.df, train_size = 0.8, stratify=data.iloc[:, -1])
        
        #print('Train AUC Score: {}'.format(roc_auc_score(train[roles['target']].values[not_nan_gpu], \
        #                                        oof_pred_gpu.data[not_nan_gpu][:, 0])))
        
    @property
    def test(self):
        #print('Преобразовываем данные')
        self.make_right_data()
        #print('Векторизуем')
        X = pd.DataFrame(np.stack([self.vectorize_sum(text, self.embeddings_pretrained) for text in self.data.iloc[:, 1]]))

        x_test = pd.concat([self.data.iloc[ :, [2, 3, 4, 5]], X], axis=1)

        with open('lib/model.pkl', 'rb') as f:
            model = pickle.load(f)
        #print('Прогнозируем')
        y_pred = model.predict_proba(x_test)[:,1]
        #print('гуд бай!')
        return y_pred

def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
