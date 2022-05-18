from typing import Tuple, Union
import pandas as pd

class task1():
    
    def __init__(self, df):
        self.df = df
        
    @property
    def train(self):

        train, test = train_test_split(self.df, train_size = 0.8, stratify=data.iloc[:, -1])
        

        
        #print('Train AUC Score: {}'.format(roc_auc_score(train[roles['target']].values[not_nan_gpu], \
        #                                        oof_pred_gpu.data[not_nan_gpu][:, 0])))
        
    @property
    def test(self):
        
        #automl2 = torch.load(f"/content/gdrive/MyDrive/Avito_weights.pth")
        #pred = automl_gpu.predict(self.df)
        #prediction = pd.DataFrame( {'prediction': list(pred.data[:, 0])}, index = range(len(df.index)))
        print('hello!')
        #return prediction['prediction']

def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
